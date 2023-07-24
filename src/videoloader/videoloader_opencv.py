import cv2
from src.videoloader.base import VideoLoaderBase


class VideoLoaderOpenCV(VideoLoaderBase):
    def __init__(self, file_path: str):
        self.video = None
        self.file_path = file_path

    def get_frame(self):
        if self.video is None:
            raise Exception("No video is currently open")

        ret, frame = self.video.read()
        if not ret:
            return None

        return frame

    def get_fps(self):
        if self.video is None:
            raise Exception("No video is currently open")

        return self.video.get(cv2.CAP_PROP_FPS)

    def is_opened(self):
        return self.video.isOpened()

    def open(self):
        self.close()

        self.video = cv2.VideoCapture(self.file_path)

        if not self.video.isOpened():
            raise Exception(f"Failed to open video: {self.file_path}")

        print(f"Video {self.file_path} is successfully opened.")

    def close(self):
        if self.video is not None:
            self.video.release()
            self.video = None
            print("Video is successfully closed.")
