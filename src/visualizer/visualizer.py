import cv2
import numpy as np
import pandas as pd
import cv2
import csv

from src.tracker.tracker_manager import TrackerManager
from src.videoloader.base import VideoLoaderBase
from src.visualizer.base import VisualizerBase


class Visualizer(VisualizerBase):
    def __init__(self,
                 tracker_manager: TrackerManager,
                 video_loader: VideoLoaderBase,
                 roi_percent: float,
                 log_path: str,
                 output_path: str,
                 custom_id: bool = False):
        self.tracker_manager = tracker_manager
        self.video_loader = video_loader
        self.roi_percent = roi_percent / 100
        self.bbox_center = None
        self.bbox = None
        self.frame_counter = 0
        self.custom_id = custom_id

        self.log_path = log_path
        self.output_path = output_path
        self.log_file = open(self.log_path, "w", newline='')
        self.writer = csv.writer(self.log_file)
        self.writer.writerow(['Frame', 'Event', 'X', 'Y', 'Width', 'Height'])  # Column headers

        self.video_writer = None

        self.video_loader.open()
        self.initial_frame = self.video_loader.get_frame()
        self.frame = self.initial_frame.copy()

        cv2.namedWindow("Tracking window")
        cv2.setMouseCallback("Tracking window", self.mouse_click_action)

        # If custom_id is True, use selectROI to set initial bounding box
        if self.custom_id:
            self.bbox = cv2.selectROI("Tracking window", self.frame, False, False)
            self.tracker_manager.set_bbox(self.bbox)
            self.add_bbox_to_frame()

    def mouse_click_action(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.custom_id:
            self.bbox_center = (x, y)
            bbox_dim = (int(self.frame.shape[1] * self.roi_percent), int(self.frame.shape[0] * self.roi_percent))

            # Ensure bounding box is within frame
            x = max(bbox_dim[0] // 2, min(x, self.frame.shape[1] - bbox_dim[0] // 2))
            y = max(bbox_dim[1] // 2, min(y, self.frame.shape[0] - bbox_dim[1] // 2))

            self.bbox = (x - bbox_dim[0] // 2, y - bbox_dim[1] // 2, bbox_dim[0], bbox_dim[1])
            self.tracker_manager.set_bbox(self.bbox)

            self.add_bbox_to_frame()
            cv2.imshow("Tracking window", self.frame)  # update frame immediately

            # log the click event
            self.log_event('Click', self.bbox)

    def add_bbox_to_frame(self):
        if self.bbox is not None:
            x, y, w, h = map(int, self.bbox)
            print(f"{self.frame_counter}. bounding box: {x/ self.frame.shape[1]},"
                  f" {y/ self.frame.shape[0]}, {w/ self.frame.shape[1]}, {h/ self.frame.shape[0]}")
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def log_event(self, event, bbox):
        x, y, w, h = bbox
        normalized_bbox = [x / self.frame.shape[1], y / self.frame.shape[0], w / self.frame.shape[1],
                           h / self.frame.shape[0]]
        self.writer.writerow([self.frame_counter, event, *normalized_bbox])

    def reset_frame(self):
        self.bbox = None
        self.frame = self.initial_frame.copy()

    def start_video_writer(self):
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.video_loader.get_fps()
            frame_size = (self.frame.shape[1], self.frame.shape[0])
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)

    def visualize(self):
        while True:
            if self.frame is None:
                break

            cv2.imshow("Tracking window", self.frame)

            if self.custom_id:
                self.start_video_writer()
                self.begin_tracking()
                return True

            if self.bbox is not None:
                key = input("Start tracking (press Q to quit)? (Y/N/Q): ").upper()

                if key == "Y":
                    self.start_video_writer()
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
            self.frame_counter += 1

            if self.bbox is not None:
                bbox = self.tracker_manager.track(self.frame)
                if bbox is not None:
                    self.bbox = bbox
                    self.add_bbox_to_frame()
                    self.log_event('Track', self.bbox)  # log tracking event

            if self.video_writer is not None:
                self.video_writer.write(self.frame)

            cv2.imshow("Tracking window", self.frame)

            if cv2.waitKey(1) == ord('q'):
                break

        self.video_loader.close()
        self.log_file.close()
        if self.video_writer is not None:
            self.video_writer.release()
