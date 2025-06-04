# BlazePoze: Human Pose Classification Pipeline
   
This project implements a complete machine learning pipeline to classify human poses based on 3D landmark sequences. The pipeline supports data preprocessing, augmentation, training-ready dataset generation, and visualization for pose recognition tasks.

## 🚀 Features

- Custom TensorFlow data pipeline
- Augmentation with noise, scaling, and shifting
- PCA visualization of pose clusters
- Configurable and modular design

---

## 📦 Installation
   
1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`
4. Install requirements: `pip install -r requirements.txt`
   
## 🧪 Run the Pipeline

```python
from src.blazepoze.pipeline import pose_dataset

pipeline = object_pipeline.PoseDatasetPipeline(
   data_dir="data",
   sequence_length=50,
   landmarks_dim=99,
   batch_size=32,
   augmentation_config={
      "noise": [True, 0.5],
      "scale": [True, 0.3],
      "shift": [True, 0.2]
   }
)

pipeline.load_data()
pipeline.split_data()
pipeline.plot_sample(augment=True, save_fig=True)
```
