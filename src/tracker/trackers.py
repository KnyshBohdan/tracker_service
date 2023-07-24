"""
File what contain tracker implementations

There are many copy&paste here because using trackers in OpenCV is quiet the same for all types of tracker
The reason why I am broke DRY principle, because we cannot change the tracking function at the same time for
all types of trackers, because each has its own tracking principle. Some look at a part of the image, others
at the whole image

Also function initialization can be added separate from tracking function for to make the functions more understandable

TODO: find out correct approach
"""

from src.tracker.base import TrackerBase
import cv2
import numpy as np


###################
# MIL
###################

class MILTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
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
            return self.bbox

        success, bbox = self.tracker.update(frame)
        if success:
            return bbox
        else:
            self.is_initialized = False
            return None

    def set_bbox(self, bbox: np.ndarray):
        self.bbox = bbox
        self.is_initialized = False


###################
# BOOSTING
###################

class BoostingTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.legacy.TrackerBoosting_create()
        self.is_initialized = False

    def track(self, frame):
        """
        TODO: write description
        See: https://ieeexplore.ieee.org/document/5459285

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True
            return self.bbox

        success, bbox = self.tracker.update(frame)
        return bbox if success else None

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.is_initialized = False


###################
# MEDIANFLOW
###################

class MedianFlowTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.legacy.TrackerMedianFlow_create()
        self.is_initialized = False

    def track(self, frame):
        """
        TODO: write description
        See: https://iajit.org/PDF/Vol%2017,%20No.%202/16021.pdf

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True
            return self.bbox

        success, bbox = self.tracker.update(frame)

        return bbox if success else None

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.is_initialized = False


###################
# TLD
###################

class TLDTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.legacy.TrackerTLD_create()
        self.is_initialized = False

    def track(self, frame):
        """
        TODO: write description
        See: http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True
            return self.bbox

        success, bbox = self.tracker.update(frame)
        return bbox if success else None

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.is_initialized = False


###################
# KCF
###################

class KCFTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.TrackerKCF_create()
        self.is_initialized = False

    def track(self, frame):
        """
        TODO: write description
        See: https://cw.fel.cvut.cz/b182/courses/mpv/labs/4_tracking/4b_tracking_kcf

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True
            return self.bbox

        success, bbox = self.tracker.update(frame)
        return bbox if success else None

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.is_initialized = False


###################
# GOTURN
###################

class GOTURNTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.TrackerGOTURN_create()
        self.is_initialized = False

    def track(self, frame):
        """
        TODO: write description
        See: https://learnopencv.com/goturn-deep-learning-based-object-tracking/

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True
            return self.bbox

        success, bbox = self.tracker.update(frame)
        return bbox if success else None

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.is_initialized = False


###################
# MOSSE
###################

class MOSSETracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.is_initialized = False

    def track(self, frame):
        """
        TODO: write description
        See: https://api.mountainscholar.org/server/api/core/bitstreams/9ea405d8-7a84-408d-b18b-c6bcbb305e2e/content

        :param frame: The frame in which to track the object.
        :return: New bounding box (x, y, w, h) where the object is found in the frame. Return None if object is not found.
        """
        if not self.is_initialized:
            self.tracker.init(frame, self.bbox)
            self.is_initialized = True
            return self.bbox

        success, bbox = self.tracker.update(frame)
        return bbox if success else None

    def set_bbox(self, bbox):
        self.bbox = bbox
        self.is_initialized = False
