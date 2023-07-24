import cv2
from src.videoloader.base import VideoLoaderBase


class VideoLoaderOpenCV(VideoLoaderBase):
    def __init__(self):
        self.video = None

    def get_frame(self):
        if self.video is None:
            raise Exception("No video is currently open")

        ret, frame = self.video.read()
        if not ret:
            raise Exception("Failed to read frame from video")

        return frame

    def is_opened(self):
        return self.video.isOpened()

    def open(self, file_path):
        self.video = cv2.VideoCapture(file_path)

        if not self.video.isOpened():
            raise Exception(f"Failed to open video: {file_path}")

        print(f"Video {file_path} is successfully opened.")

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
            print("Video is successfully closed.")
