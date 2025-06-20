# BlazePoze Action Recognition

BlazePoze is a lightweight pipeline for classifying human actions from real-time 3D pose data streamed by a Luxonis OAK-D camera. The project combines on-device landmark detection with a Temporal Convolutional Network (TCN) running on the host CPU.

## Features

- DepthAI-based BlazePose detector for on-device 3D landmarks
- TensorFlow/Keras TCN classifier for recognizing actions
- Data pipeline with optional augmentation utilities
- Training and live inference scripts for easy experimentation

## Final Project Structure

```
blazepoze-detection/
├── data/                  # raw CSV training data
├── saved_models/          # trained `.keras` models
├── src/
│   └── blazepoze/
│       ├── data_processing/
│       │   └── pose_dataset_pipeline.py
│       ├── models/
│       │   └── tcn.py
│       ├── inference/
│       │   └── pose_action_classifier.py
│       └── utils/
│           ├── augment.py
│           ├── validation.py
│           └── logging_utils.py
├── scripts/
│   ├── train.py
│   └── run_inference.py
└── labels.txt
```

## Setup and Installation

```bash
# clone the repository
$ git clone <repo-url>
$ cd blazepoze-detection

# create and activate a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate

# install dependencies
$ pip install -r requirements.txt
```

## Usage

### Training

Train a TCN model on your dataset:

```bash
python scripts/train.py --data-dir ./data/ --output-dir ./saved_models/
```

- `--data-dir` Path containing label subfolders with pose CSV files
- `--output-dir` Directory where the trained `.keras` model will be saved

### Live Inference

Run the real-time pipeline with an OAK-D camera:

```bash
python scripts/run_inference.py \
    --keras_model ./saved_models/pose_classifier_oak.keras \
    --pd_model depthai_blazepose/models/pose_detection_sh4.blob \
    --lm_model depthai_blazepose/models/pose_landmark_full_sh4.blob
```

- `--keras_model` Trained TCN model
- `--pd_model` BlazePose pose detection blob
- `--lm_model` BlazePose landmark blob
- `--label_file` Optional text file with class labels

## Model Architecture

1. **Pose Detector** – BlazePose model running on the OAK-D device.
2. **Landmark Estimator** – Generates 3D landmarks for each frame on device.
3. **TCN Classifier** – Temporal Convolutional Network implemented in TensorFlow/Keras. It processes the landmark sequence on the host CPU to classify the current action.

## License

This project is licensed under the MIT License.
