from pytubefix import YouTube
import nltk
from nltk.tokenize import sent_tokenize
from webvtt import WebVTT
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse
import pandas as pd

from face_detector import get_facial_landmarks
import tqdm as tqdm

# Download the video from the URL
# Save the video to the data_original directory, with the hash(title) as the filename
def download_video(video_url, output_dir="data_original"):
    yt = YouTube(video_url)
    title = hash(yt.title)
    
    yt.streams.get_highest_resolution().download(output_dir, filename = f"{title}.mp4")
    return title

# Parse the transcript of the video and return a list of segments
# Each segment is a dictionary with the following keys:
# - start: The start time of the segment
# - duration: The duration of the segment
# - sentence: The text of the segment
def parse_transcript(video_url):
    video_id = video_url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    # Iterate through the transcript and replace \u00a0 with a space for the "text" key
    for i in range(len(transcript)):
        transcript[i]["text"] = transcript[i]["text"].replace("\u00a0", " ").replace("\n", " ")

    segments = []

    for i in range(len(transcript)):
        text = transcript[i]["text"]
        start_time = transcript[i]["start"]
        duration = transcript[i]["duration"]

        segments.append({
            "start": start_time,
            "duration": duration,
            "sentence": text
        })

    return segments

# Split the video into segments based on the transcript
# Save each segment to the output directory with the hash(title) as the filename
# Save the text of each segment to a text file with the same name as the video file
# Split the csv file into segments based on the transcript. 
# Do it based on the "timestap" column.
def split_video(video_path, segments, title):
    video = VideoFileClip(video_path)

    pandas_df = pd.read_csv(f"data_processed/{title}_face_blendshapes.csv")
    for idx, segment in enumerate(segments, start=1):
        start_time = segment["start"]
        duration = segment["duration"]
        end_time = start_time + duration
        output_file = f"src/output/segment_{hash(title)}_{idx}.mp4"
        video_segment = video.subclip(start_time, end_time)
        video_segment.write_videofile(output_file)

        with(open(f"src/input/segment_{hash(title)}_{idx}.txt", "w")) as f:
            f.write(segment["sentence"])

        segment_df = pandas_df[(pandas_df["timestamp"] >= start_time) & (pandas_df["timestamp"] <= end_time)]
        segment_df.to_csv(f"src/output/segment_{hash(title)}_{idx}.csv", index=False)

    video.close()

def main(video_url):
    print("DOWNLOADING VIDEO")
    title = download_video(video_url)
    print("DOWNLOAD COMPLETE")

    original_video_path = f"./data_original/{title}.mp4"

    print("DETECTING FACES")
    get_facial_landmarks(original_video_path, f"./data_processed/{title}.mp4")
    print("FACE DETECTION COMPLETE")

    processed_video_path = f"./data_processed/{title}.mp4"

    print("PARSING TRANSCRIPT")
    segments = parse_transcript(video_url)
    print("PARSING COMPLETE")
    
    print("SPLITTING VIDEO")
    split_video(processed_video_path, segments, title)
    print("SPLITTING COMPLETE")

    print("-------------DONE-------------")

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Download and split a YouTube video based on its transcript")
    parser.add_argument("--v", type=str, help="The URL of the YouTube video to process")
    args = parser.parse_args()

    if args.v:
        main(args.v)
