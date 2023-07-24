import numpy as np

from src.tracker.tracker_manager import TrackerManager
from src.videoloader import VideoLoaderOpenCV
from src.tracker.trackers import MILTracker, BoostingTracker
from src.visualizer.visualizer import Visualizer

PATH_TO_VIDEO = "data/test_data/test.mp4"

if __name__ == "__main__":
    loader = VideoLoaderOpenCV(PATH_TO_VIDEO)

    tracker_manager = TrackerManager(BoostingTracker(np.ndarray([0, 0, 0, 0])))

    visualizer = Visualizer(tracker_manager=tracker_manager,
                            video_loader=loader,
                            roi_percent=50,
                            log_path="output/log.csv",
                            output_path="output/output.mp4",
                            custom_id=False)

    visualizer.visualize()
