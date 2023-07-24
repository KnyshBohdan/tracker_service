from abc import ABC, abstractmethod


class TrackerBase(ABC):
    def __init__(self, bbox):
        self.bbox = bbox

    def set_bbox(self, bbox):
        self.bbox = bbox

    @abstractmethod
    def track(self, frame):
        """
        This function should implement the logic for tracking the object in the provided frame.
        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        pass
