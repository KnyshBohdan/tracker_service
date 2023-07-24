import pytest
import cv2
import numpy as np

from src.videoloader.videoloader_opencv import VideoLoaderOpenCV
from tests.cache import TEST_VIDEO


class TestVideoLoader:
    def setup_method(self):
        self.video_loader = VideoLoaderOpenCV()

    def teardown_method(self):
        self.video_loader.close()

    def test_open_close(self):
        self.video_loader.open(TEST_VIDEO)
        assert self.video_loader.video is not None
        assert self.video_loader.is_opened() == True

        self.video_loader.close()
        assert self.video_loader.video is None

    def test_get_frame(self):
        self.video_loader.open(TEST_VIDEO)

        frame = self.video_loader.get_frame()

        assert isinstance(frame, np.ndarray)

        self.video_loader.close()
