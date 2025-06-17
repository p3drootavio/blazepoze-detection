# BlazePoze: Human Pose Classification Pipeline

BlazePoze provides a full workflow for preparing 3D pose datasets, training Temporal Convolutional Networks (TCNs) and deploying the resulting model to Luxonis OAK cameras. The library includes data augmentation utilities, visualization helpers and scripts for exporting models.

## Features

- TensorFlow data pipeline for sequences of 3D landmarks
- Augmentation functions (Gaussian noise, scaling and shifting)
- Pose visualization using PCA and t-SNE
- TCN architectures for classification or regression
- Export to ONNX and DepthAI `.blob`

## Installation

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/macOS: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. (Optional) Install the project in editable mode: `pip install -e .`

## Repository Structure

- `src/blazepoze/` – library code (datasets, models, utils)
- `scripts/` – training and conversion scripts
- `models/` – pre-trained (`pretrained/`) and exported (`exported/`) models

## Dataset Format

The training pipeline expects a folder of CSV files organised by label:

```
<data_dir>/
└── <label>/
    ├── sample_d__landmarks.csv
    ├── sample_p__landmarks.csv
    └── ...
```

Each CSV contains 50 frames with 99 landmark values per frame.

## Usage

### Training
Edit the paths inside `scripts/train_pipeline.py` and run:

```bash
python scripts/train_pipeline.py
```

### Converting to ONNX
After training a Keras model, export it:

```bash
python scripts/convert_to_onnx.py
```

To convert the ONNX model to a DepthAI blob use:

```bash
python scripts/onnx_to_blob.py path/to/model.onnx
```

### DepthAI Demo
Connect an OAK camera and run the demo by providing paths to both the
gesture classifier and BlazePose blobs:

```bash
python scripts/run_depthai.py --classifier-blob models/deployed/blazepose.blob \
                             --pose-blob path/to/blazepose.blob
```

### Simplified DepthAI Demo
For a minimal pipeline that only requires the classifier blob you can run:

```bash
python scripts/run_depthai_simplified.py --classifier-blob models/deployed/blazepose.blob
```

## Additional Utilities

- `scripts/verify_data.py` – visualise CSV sequences and create a video
- `src/blazepoze/visualization/pose_visualizer.py` – PCA, t-SNE and time-series plots

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
