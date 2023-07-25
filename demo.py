import argparse
import numpy as np

from src.tracker.tracker_manager import TrackerManager
from src.videoloader import VideoLoaderOpenCV
from src.tracker.trackers import MILTracker, BoostingTracker, TLDTracker, KCFTracker,\
    MOSSETracker, GOTURNTracker, MedianFlowTracker
from src.visualizer.visualizer import Visualizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="A program to visualize tracking in a video")

    parser.add_argument("--tracker", type=str, required=True,
                        help="Type of tracker to be used")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--output_file", type=str, default="output.mp4",
                        help="Path to the output video file")
    parser.add_argument("--log_file", type=str, default="log.csv",
                        help="Path to the log file")
    parser.add_argument("--roi_percent", type=float, default=50,
                        help="Percentage of the ROI to be used for tracking")
    parser.add_argument("--custom_roi", action="store_true",
                        help="Use custom ROI instead of static size")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    loader = VideoLoaderOpenCV(args.input_file)

    tracker_manager = TrackerManager()

    # can be done using factory, but it will be overkill
    if args.tracker == "BOOSTING":
        tracker_manager.set_tracker(BoostingTracker())
    elif args.tracker == "MIL":
        tracker_manager.set_tracker(MILTracker())
    elif args.tracker == "TLD":
        tracker_manager.set_tracker(TLDTracker())
    elif args.tracker == "KCF":
        tracker_manager.set_tracker(KCFTracker())
    elif args.tracker == "GOTURN":
        tracker_manager.set_tracker(GOTURNTracker())
    elif args.tracker == "MOSSE":
        tracker_manager.set_tracker(MOSSETracker())
    elif args.tracker == "MEDIANFLOW":
        tracker_manager.set_tracker(MedianFlowTracker())

    visualizer = Visualizer(tracker_manager=tracker_manager,
                            video_loader=loader,
                            roi_percent=args.roi_percent,
                            log_path=args.log_file,
                            output_path=args.output_file,
                            custom_id=args.custom_roi)

    visualizer.visualize()
