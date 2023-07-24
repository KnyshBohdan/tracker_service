# Tracker Service

Tracker Service is a robust Python application that allows users to compare the performances of various tracking algorithms on a given video input. It supports several popular trackers such as BOOSTING, MIL, TLD, KCF, MOSSE, and MEDIANFLOW.

For a quick demonstration of the service, check out the demo script demo.py

## Prerequisites

- Python 3.10 or higher
- `virtualenv` (optional, but recommended)

## Installation

For Windows

```bash
git clone https://github.com/KnyshBohdan/tracker_service
cd tracker_service
mkdir "output"
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

For GNU\Linux

```bash
git clone https://github.com/KnyshBohdan/tracker_service
cd tracker_service
mkdir "output"
python3 -m venv env
env\Scripts\activate.bat
source env/bin/activate
pip install -r requirements.txt
```

## Install conda

```bash
git clone https://github.com/KnyshBohdan/tracker_service
cd tracker_service
conda create --name env python=3.10
conda activate env
pip install -r requirements.txt
```

# Test

To ensure the application runs as expected, run the following command:

```bash
pytest
```


# Demo

To demonstrate the tracker service in action, use the following command:

```bash
python demo.py --tracker <tracker> --input_file <input_file> --output_file <output_file> --log_file <log_file> --roi_percent <roi_percent> --custom_roi
```

Replace the variables with your preferences:

`<tracker>` - Algorithm to track. Supported trackers include:
* BOOSTING
* MIL
* TLD
* KCF
* MOSSE
* MEDIANFLOW

`<input_file>` - Path to the input video file.

`<output_file>` - Path to the output video file.

`<log_file>` - Path to the log file.

`<roi_percent> `- Percentage of the ROI (Region of Interest) to be used for tracking.

`custom_roi` - Use custom ROI instead of static size. It is a flag to determine whether to use a custom ROI.

Example:

```bash
python demo.py --tracker MEDIANFLOW --input_file data/test_data/Video_test_CV_10sec.mp4 --output_file output/output.mp4 --log_file output/log.csv --roi_percent 5 --custom_roi
```

Please ensure you provide the appropriate paths for your input, output, and log files for the command to run successfully.