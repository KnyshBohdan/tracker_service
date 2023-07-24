import cv2
import numpy as np

from src.tracker.tracker_manager import TrackerManager
from src.videoloader.base import VideoLoaderBase


class Visualizer:
    def __init__(self, tracker_manager, video_loader, roi_percent):
        self.tracker_manager = tracker_manager
        self.video_loader = video_loader
        self.roi_percent = roi_percent / 100
        self.bbox_center = None
        self.bbox = None

        self.video_loader.open()
        self.initial_frame = self.video_loader.get_frame()
        self.frame = self.initial_frame.copy()

        cv2.namedWindow("Tracking window")
        cv2.setMouseCallback("Tracking window", self.mouse_click_action)

    def mouse_click_action(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bbox_center = (x, y)
            bbox_dim = (int(self.frame.shape[1] * self.roi_percent), int(self.frame.shape[0] * self.roi_percent))
            self.bbox = (x - bbox_dim[0] // 2, y - bbox_dim[1] // 2, bbox_dim[0], bbox_dim[1])
            self.tracker_manager.set_bbox(self.bbox)

            self.add_bbox_to_frame()
            cv2.imshow("Tracking window", self.frame)  # update frame immediately

    def add_bbox_to_frame(self):
        if self.bbox is not None:
            x, y, w, h = map(int, self.bbox)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def reset_frame(self):
        self.bbox = None
        self.frame = self.initial_frame.copy()

    def visualize(self):
        while True:
            if self.frame is None:
                break

            cv2.imshow("Tracking window", self.frame)

            if self.bbox is not None:
                key = input("Start tracking (press Q to quit)? (Y/N): ").upper()

                if key == "Y":
                    self.begin_tracking()
                elif key == "N":
                    self.reset_frame()
                elif key == "Q":
                    return False

            if cv2.waitKey(1) == ord('q'):
                return False

    def begin_tracking(self):
        while True:
            self.frame = self.video_loader.get_frame()
            if self.frame is None:
                break

            if self.bbox is not None:
                bbox = self.tracker_manager.track(self.frame)
                if bbox is not None:
                    self.bbox = bbox
                    self.add_bbox_to_frame()

            cv2.imshow("Tracking window", self.frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.video_loader.close()
