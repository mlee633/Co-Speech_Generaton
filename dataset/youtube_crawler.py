from pytubefix import YouTube
import nltk
from nltk.tokenize import sent_tokenize
from webvtt import WebVTT
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse

from face_detector import get_facial_landmarks
import tqdm as tqdm

def parse_transcript(video_url):
    video_id = video_url.split("v=")[1]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    # Iterate through the transcript and replace \u00a0 with a space for the "text" key
    for i in range(len(transcript)):
        transcript[i]["text"] = transcript[i]["text"].replace("\u00a0", " ").replace("\n", " ")

    return transcript

def download_video(video_url, output_dir="video_original"):
    yt = YouTube(video_url)
    title = yt.title
    
    yt.streams.get_highest_resolution().download(output_dir)
    return title

def split_video(video_path, segments, title):
    video = VideoFileClip(video_path)
    for idx, segment in enumerate(segments, start=1):
        start_time = segment["start"]
        duration = segment["duration"]
        end_time = start_time + duration
        output_file = f"output/segment_{hash(title)}_{idx}.mp4"
        video_segment = video.subclip(start_time, end_time)
        video_segment.write_videofile(output_file)

        with(open(f"input/segment_{hash(title)}_{idx}.txt", "w")) as f:
            f.write(segment["sentence"])

    video.close()

def main(video_url):
    print("DOWNLOADING VIDEO")
    title = download_video(video_url)
    print("DOWNLOAD COMPLETE")

    video_path = f"./video_original/{title}.mp4"

    print("DETECTING FACES")
    get_facial_landmarks(video_path, f"./video_processed/{title}.mp4")
    print("FACE DETECTION COMPLETE")

    video_path = f"./video_processed/{title}.mp4"

    print("PARSING TRANSCRIPT")
    transcript = parse_transcript(video_url)
    print("PARSING COMPLETE")
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

    print("SPLITTING VIDEO")
    split_video(video_path, segments, title)
    print("SPLITTING COMPLETE")

    print("-------------DONE-------------")

if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Download and split a YouTube video based on its transcript")
    parser.add_argument("--v", type=str, help="The URL of the YouTube video to process")
    args = parser.parse_args()

    if args.v:
        main(args.v)
