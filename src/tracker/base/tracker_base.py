from abc import ABC, abstractmethod
import numpy as np


class TrackerBase(ABC):
    def __init__(self, bbox: np.ndarray):
        self.bbox = bbox

    def set_bbox(self, bbox: np.ndarray):
        self.bbox = bbox

    @abstractmethod
    def track(self, frame: np.ndarray) -> np.ndarray:
        """
        This function should implement the logic for tracking the object in the provided frame.
        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        pass
