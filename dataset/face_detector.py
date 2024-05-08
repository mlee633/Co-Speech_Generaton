import cv2
import numpy as np
import tqdm as tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import time

def crop_face_from_video(video_url):
    # Load the pre-trained face detector and facial landmark detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input video
    video = cv2.VideoCapture(video_url)

    # Get the video properties
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to save the cropped video
    output_video = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Process each frame of the video
    for _ in tqdm.tqdm(range(total_frames)):
        ret, frame = video.read()
        
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Crop each detected face and save it
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            output_video.write(face)        

    # Release the video objects
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

# Detect face and facial landmarks
# Remove background and only remain the face (landmarks)
# Output will be the video with only the face
# Use mediapipe for face detection and facial landmarks, openCV for video processing
def get_facial_landmarks(video_path, output_video_path):
    model_path = "./face_landmarker.task"

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options = BaseOptions(model_asset_path = model_path),
        running_mode = VisionRunningMode.VIDEO,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        video = cv2.VideoCapture(video_path)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        for i in tqdm.tqdm(range(3600)):
            ret, frame = video.read()
            curr_time = mp.Timestamp.from_seconds(time.time()).value
            
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
            face_landmarker_result = landmarker.detect_for_video(mp_image, curr_time)

            # Black screen to remove background
            frame = np.zeros(frame.shape, dtype=np.uint8)
            frame = draw_landmarks_on_image(frame, face_landmarker_result)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_video.write(frame)

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


if __name__ == "__main__":
    get_facial_landmarks("output\segment_6056801418719529898_1.mp4")