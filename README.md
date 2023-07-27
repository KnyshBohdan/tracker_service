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
env/Scripts/activate
pip install -r requirements.txt
```

For GNU\Linux

```bash
git clone https://github.com/KnyshBohdan/tracker_service
cd tracker_service
mkdir "output"
python3 -m venv env
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

Can make error while working with GOTURN tracker, more about it in GOTURN

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
* CSRT
* MEDIANFLOW
* GOTURN

`<input_file>` - Path to the input video file.

`<output_file>` - Path to the output video file.

`<log_file>` - Path to the log file.

`<roi_percent> `- Percentage of the ROI (Region of Interest) to be used for tracking.

`custom_roi` - Use custom ROI instead of static size. It is a flag to determine whether to use a custom ROI.

Example:

```bash
python demo.py --tracker MEDIANFLOW --input_file tests/test_data/test.mp4 --output_file output/output.mp4 --log_file output/log.csv --roi_percent 5
```

Please ensure you provide the appropriate paths for your input, output, and log files for the command to run successfully.

### GOTURN

GOTURN, short for Generic Object Tracking Using Regression Networks, is a Deep Learning based tracking algorithm.

To work with it, run next code:

```bash
git clone https://github.com/spmallick/goturn-files
cd goturn-files
cat goturn.caffemodel.zip* > goturn.caffemodel.zip
unzip goturn.caffemodel.zip
mv goturn.caffemodel ../goturn.caffemodel
mv goturn.prototxt ../goturn.prototxt
```

## Docker

Docker Workflow

In order to work within a Docker environment, we first need to construct our Docker image. You can create the image using the Dockerfile provided by running the following command:

```bash
 docker build -t tracker-service .
```

Given the need to interface with OpenCV's graphical user interface, we've created a script that runs a Docker container. This container can interact with the host's X server, enabling the display output. This functionality is especially useful for working with OpenCV, a leading computer vision library often used for tasks such as displaying images or videos.

To run the Docker container, execute the script runDocker.sh with your video file's absolute path:

```bash
sh runDocker.sh <your_absoulute_path_to_video>
```

Please replace `<absolute_path_to_your_video>` with the actual path to your video file.

If everything was set up correctly, you'll receive a command prompt providing access to the container. You can confirm if the video file, video.mp4 (which is your video), is present within the container.

Now, you can operate as if you're working within a Python environment. For instance, you can execute the following command:

```bash
python demo.py --tracker MEDIANFLOW --input_file video.mp4 --output_file output.mp4 --log_file log.csv --roi_percent 5
```

After you've finished your work and exited the container, you can extract the output.mp4 and log.csv files by running:

```bash
docker cp <name_of_container>:log.csv .
docker cp <name_of_container>:output.mp4 .
```

In the above commands, replace `<container_name>` with the name of your Docker container. After the execution of these commands, you'll find the output.mp4 and log.csv files in your current directory.