import numpy as np
import pandas as pd
import tensorflow as tf
from src.blazepoze.pipeline.pose_dataset import PoseDatasetPipeline


def _create_sample_csv(path):
    data = np.random.rand(50, 100)  # 99 features + index column
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_load_and_dataset(tmp_path):
    label_dir = tmp_path / "label"
    label_dir.mkdir()

    _create_sample_csv(label_dir / "sample1_d__landmarks.csv")
    _create_sample_csv(label_dir / "sample2_d__landmarks.csv")
    _create_sample_csv(label_dir / "sample1_p__landmarks.csv")
    _create_sample_csv(label_dir / "sample2_p__landmarks.csv")

    # Print debug info
    for file in label_dir.iterdir():
        print(f"{file.name}: {pd.read_csv(file).shape}")

    config = {"noise": [False, 0.0], "scale": [False, 0.0], "shift": [False, 0.0]}
    pipeline = PoseDatasetPipeline(str(tmp_path), 50, 99, 8, config)
    pipeline.load_data()

    assert pipeline.X.shape == (4, 50, 99)
    assert len(pipeline.y) == 4

    pipeline.split_data(test_size=0.5, valid_size=0.5)
    ds = pipeline.get_tf_dataset("train")
    for batch_x, batch_y in ds.take(1):
        assert batch_x.shape[1:] == (50, 99)
        assert batch_y.shape[0] > 0
