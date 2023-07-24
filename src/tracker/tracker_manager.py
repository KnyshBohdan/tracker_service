"""
Manager class for postprocess and error handling
For simple case, can be deleted
"""

from src.tracker.base import TrackerBase
import numpy as np


class TrackerManager:
    def __init__(self,
                 tracker: TrackerBase):
        self.tracker = tracker

    def set_bbox(self, bbox):
        self.tracker.set_bbox(bbox)

    def track(self, frame):
        if not isinstance(frame, np.ndarray):
            raise ValueError("Frame has to have numpy.ndarray type")

        if frame.size == 0:
            raise ValueError("Frame is empty")

        try:
            return self.tracker.track(frame)
        # dummy error handling
        except Exception as e:
            raise e
