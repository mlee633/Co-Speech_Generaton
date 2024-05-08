import cv2

def save_first_10_seconds(video_path, output_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of frames for the first 10 seconds
    target_frames = int(fps * 10)

    # Create a VideoWriter object to save the extracted clip
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Read and write the frames for the first 10 seconds
    for frame_number in range(target_frames):
        ret, frame = video.read()
        if not ret:
            break
        writer.write(frame)

    # Release the video file and writer
    video.release()
    writer.release()

# Example usage
video_path = "dataset/data_original/-4681178808489471002.mp4"
output_path = "dataset/data_original/-4681178808489471001.mp4"
save_first_10_seconds(video_path, output_path)