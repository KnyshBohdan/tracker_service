"""
File what contain tracker implementations
"""

from src.tracker.base import TrackerBase
import cv2
import numpy as np


class MILTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray):
        super().__init__(bbox)
        self.tracker = cv2.TrackerMIL_create()
        self.is_initialized = False

    def track(self, frame: np.ndarray) -> np.ndarray:
        """
        Multiple Instance Learning (MIL) algorithm for tracking objects in video frames.
        This tracking method is robust to variations in the object's appearance, and it
        updates a model of the object over time for improved tracking. If the object is
        not detected in the frame, the method returns None.
        See: https://faculty.ucmerced.edu/mhyang/papers/cvpr09a.pdf

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True

        success, bbox = self.tracker.update(frame)
        if success:
            return bbox
        else:
            self.is_initialized = False
            return None

    def set_bbox(self, bbox: np.ndarray):
        self.bbox = bbox
        self.is_initialized = False
