# BlazePoze Action Recognition

BlazePoze is a streamlined pipeline for recognizing human actions from 3D pose data captured by a Luxonis OAK-D camera. The camera runs BlazePose for landmark extraction while the host CPU executes a Temporal Convolutional Network (TCN) to classify the action.

## Features

- Real-time landmark detection with DepthAI
- TensorFlow/Keras TCN model for action recognition
- Dataset pipeline with optional augmentation utilities
- Scripts for training and live inference

## Essential vs. Obsolete Files

### Essential
- `src/blazepoze/pipeline/depthai_simplified_buffer.py` – `PoseActionClassifier` inference pipeline
- `src/blazepoze/pipeline/pose_dataset.py` – `PoseDatasetPipeline` for data handling
- `src/blazepoze/pipeline/tnc_model_strong.py` – TCN model definition
- `src/blazepoze/utils/augment.py`, `validation.py`, `logging_utils.py`
- `scripts/run_depthai_buffer.py` – live inference script
- `scripts/train_pipeline_for_oak.py` – training script producing the `.keras` model
- `models/pretrained/*.keras` – trained models used for inference

### Obsolete
- `src/blazepoze/pipeline/depthai.py`, `depthai_simplified.py`
- `src/blazepoze/pipeline/tcn_model_weak.py`, `tcn_ed_model.py`
- Legacy scripts such as `run_depthai.py`, `run_depthai_simplified.py`, `train_pipeline.py`, `train_pipeline_edtcn.py`
- ONNX conversion utilities (`convert_to_onnx.py`, `onnx_to_blob.py`, `calculate_flops.py`)

## Refactored Project Layout

```
blazepoze-detection/
├── data/                  # raw training CSV files
├── saved_models/          # final .keras models
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

## Code Refinement Suggestions

- Rename `PoseActionClassifier._prepare_input_tensor` variables for clarity (e.g., `landmark_buffer` → `buffer`) and add comments explaining the reshaping logic.
- In `run_depthai_buffer.py` ensure each argument description clearly states the expected path and add a newline at the end of the file.

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

```bash
python scripts/train.py --data-dir ./data/ --output-dir ./saved_models/
```
- `--data-dir` directory with label subfolders containing pose CSV files
- `--output-dir` directory where the trained model will be saved

### Live Inference

```bash
python scripts/run_inference.py \
    --keras_model ./saved_models/pose_classifier_oak.keras \
    --pd_model depthai_blazepose/models/pose_detection_sh4.blob \
    --lm_model depthai_blazepose/models/pose_landmark_full_sh4.blob
```
- `--keras_model` path to the trained TCN model
- `--pd_model` BlazePose pose detection blob
- `--lm_model` BlazePose landmark blob
- `--label_file` optional text file with class labels

## Model Architecture

1. **Pose Detector** – BlazePose running on the OAK-D device
2. **Landmark Estimator** – produces 3D landmarks on device
3. **TCN Classifier** – host-side TensorFlow/Keras network that classifies the current action

## License

This project is licensed under the MIT License.

