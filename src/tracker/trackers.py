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
        The Boosting Tracker is a tracking method based on machine learning techniques. It works by training a classifier
        to distinguish between the object and the background in the first frame. For subsequent frames, the classifier
        is updated to continue tracking the object. The algorithm is designed to work with changes in illumination,
        object scale and object pose. However, the method can fail with abrupt motion changes, occlusions, and object rotation.

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
        The MedianFlow Tracker works by tracking the object in both forward and backward directions in time and
        estimates the object's displacement using the median of these displacements. This allows the tracker to
        detect and correct tracking failures early on, making it more robust to occlusions and other tracking
        failures. However, it may not perform well with fast-moving objects or sudden changes in object appearance.

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
        The TLD (Tracking, Learning and Detection) Tracker, also known as the Predator Tracker, is designed to be robust
        to various types of object motion including occlusion, object scaling and rotation, and camera motion. The TLD
        algorithm works by tracking the object, learning its appearance, and detecting it in each frame. If the tracker
        loses the object, the detector can find it again. This makes TLD very robust, but it may suffer from false positives
        in complex scenes.

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
        The KCF (Kernelized Correlation Filter) Tracker uses a method based on correlation filters and kernel methods.
        KCF is capable of handling changes in object appearance, such as changes in scale or aspect ratio. It is also
        quite fast, making it suitable for real-time applications. However, it can fail in cases of occlusions or drastic
        changes in object appearance.

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
        The GOTURN (Generic Object Tracking Using Regression Networks) Tracker is a deep learning-based method.
        It uses a trained Convolutional Neural Network (CNN) to predict the bounding box of the target object
        in the next frame given the current frame and the bounding box in the current frame. This method is robust
        to changes in object scale, object pose, and illumination changes, but it requires a pre-trained model
        and may not perform well for objects significantly different from those in the training data.

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
        The MOSSE (Minimum Output Sum of Squared Error) Tracker is a correlation filter-based method that is
        especially robust to changes in scale and in-plane rotation. It is also very fast, making it suitable
        for real-time applications. The MOSSE tracker works by creating a model of the object's appearance and
        updating this model over time to track the object through the video sequence. However, it may not handle
        drastic changes in object appearance or severe occlusions.

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


###################
# CSRT
###################

class CSRTTracker(TrackerBase):
    def __init__(self, bbox: np.ndarray = np.array([0, 0, 0, 0])):
        super().__init__(bbox)
        self.tracker = cv2.legacy.TrackerCSRT_create()
        self.is_initialized = False

    def track(self, frame):
        """
        The CSRT Tracker is based on the Discriminative Correlation Filter method. It combines the reliability maps
        to weight the contributions of each pixel to the filter response, enabling more accurate tracking of objects.
        It performs well in several aspects, including scale changes, rotation, and occlusion, and is particularly
        suited for tracking smaller objects. However, it may not be as fast as other methods such as KCF or MOSSE.

        See: https://arxiv.org/pdf/1611.08461.pdf

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



