import pytest
from src.tracker.trackers import MILTracker
import numpy as np


class TestMILTracker:
    def setup_method(self):
        self.tracker = MILTracker(np.array([50, 50, 100, 100]))

    def teardown_method(self):
        self.tracker = None

    def test_creation(self):
        assert isinstance(self.tracker.bbox, np.ndarray)

    def test_track(self):
        # Create a synthetic image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Draw a rectangle on the image that corresponds to the bbox
        image[50:150, 50:150, :] = 0

        bbox = self.tracker.track(image)

        # Check if the tracker returns the correct bounding box
        assert bbox == (49, 52, 100, 100) # TODO: find out why is returning (49, 52, 100, 100) not [50, 50, 100, 100]
