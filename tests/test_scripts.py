import numpy as np
from scripts import verify_data


def test_load_data_empty(tmp_path):
    assert verify_data.load_data(tmp_path) == []


def test_create_frame_images(tmp_path):
    frame_data_list = [np.random.rand(50, 99).astype(np.float32)]
    out_dir = tmp_path / "frames"
    verify_data.create_frame_images(frame_data_list, 0, out_dir)
    images = list(out_dir.glob("*.png"))
    assert len(images) == 50
