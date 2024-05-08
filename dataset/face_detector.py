import cv2
import numpy as np
import tqdm as tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import matplotlib.pyplot as plt
import pandas as pd

import time

def get_facial_landmarks(video_path, output_video_path):
    model_path = "./face_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = model_path),
        running_mode = VisionRunningMode.VIDEO,
        output_face_blendshapes = True,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        video = cv2.VideoCapture(video_path)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        # Pandas dataframe with 53 columns, representing the 52 face blendshapes categories and a timestamp
        # First row is the column names
        face_blendshapes = pd.DataFrame()

        for i in tqdm.tqdm(range(total_frames)):
            ret, frame = video.read()
            curr_time = mp.Timestamp.from_seconds(i/fps).value
            
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            face_landmarker_result = landmarker.detect_for_video(mp_image, curr_time)
            # Black screen to remove background
            frame = np.zeros(frame.shape, dtype=np.uint8)

            if len(face_landmarker_result.face_blendshapes) > 0:
                face_blendshapes = append_face_blendshape(face_landmarker_result.face_blendshapes[0], face_blendshapes, i, fps)
                frame = draw_landmarks_on_image(frame, face_landmarker_result)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_video.write(frame)

        # Save the face blendshapes to a CSV file
        title = output_video_path.split("/")[-1].split(".")[0]
        directory = output_video_path.split("/")[:-1][-1]
        face_blendshapes.to_csv(f"./{directory}/{title}_face_blendshapes.csv", index=False)

        video.release()
        output_video.release()
        cv2.destroyAllWindows()

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    option = 1
    if option == 1:
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
    elif option == 2:
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
    elif option == 3:
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
        
    return annotated_image
  
def append_face_blendshape(face_blendshapes, face_blendshape_df, frame_number, fps):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]

    # Append the timestamp to the list of face blendshapes scores.
    face_blendshapes_scores.append(frame_number / fps)

    # Create a dictionary to represent the row to be appended
    row_data = {name: score for name, score in zip(face_blendshapes_names, face_blendshapes_scores)}
    row_data["timestamp"] = frame_number / fps

    # Append the row data to the list
    face_blendshape_df = pd.concat([face_blendshape_df, pd.DataFrame(row_data, index=[0])], ignore_index=True)

    return face_blendshape_df



if __name__ == "__main__":
    get_facial_landmarks("./data_original/-4681178808489471001.mp4", "./data_processed/-4681178808489471001.mp4")