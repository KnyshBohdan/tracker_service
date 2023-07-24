import pytest
from src.tracker.trackers import MILTracker, BoostingTracker, MOSSETracker,\
    MedianFlowTracker, TLDTracker, KCFTracker, GOTURNTracker
from src.tracker.tracker_manager import TrackerManager
import numpy as np


tracker_classes = [MILTracker, BoostingTracker, TLDTracker, KCFTracker, MOSSETracker]
# TODO: solve errors in GOTURNTracker
# TODO: solve errors in MedianFlowTracker

class TestTrackers:
    @pytest.mark.parametrize("tracker_class", tracker_classes)
    def test_creation(self, tracker_class):
        tracker = tracker_class(np.array([50, 50, 100, 100]))
        assert isinstance(tracker.bbox, np.ndarray)

    @pytest.mark.parametrize("tracker_class", tracker_classes)
    def test_track(self, tracker_class):
        # Setup
        tracker = tracker_class(np.array([50, 50, 100, 100]))

        # Create a synthetic image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Draw a rectangle on the image that corresponds to the bbox
        image[50:150, 50:150, :] = 0

        # just initialization
        bbox = tracker.track(image)

        # real test
        bbox = tracker.track(image)

        # Check if the tracker returns the correct bounding box
        assert len(bbox) == 4


class TestTrackerManager:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tracker_manager = TrackerManager(MILTracker(np.array([50, 50, 100, 100])))

    def test_set_bbox(self):
        bbox = np.array([1, 2, 3, 4])
        self.tracker_manager.set_bbox(bbox)
        assert np.array_equal(self.tracker_manager.tracker.bbox, bbox)

    def test_track(self):
        # Create a synthetic image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255

        # Draw a rectangle on the image that corresponds to the bbox
        image[50:150, 50:150, :] = 0

        bbox = self.tracker_manager.track(image)

        # Check if the tracker returns the correct bounding box
        assert (bbox == (50, 50, 100, 100)).all()

    def test_track_wrong_type(self):
        # Pass a non-numpy.ndarray object
        with pytest.raises(ValueError, match="Frame has to have numpy.ndarray type"):
            self.tracker_manager.track("not an ndarray")

    def test_track_empty_frame(self):
        # Pass an empty frame
        with pytest.raises(ValueError, match="Frame is empty"):
            self.tracker_manager.track(np.array([[], [], []]))