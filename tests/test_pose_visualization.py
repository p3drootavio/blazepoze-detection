import matplotlib
matplotlib.use("Agg")

from scripts import pose_visualization


def test_plot_predictions(tmp_path):
    preds = [
        (0.0, 0, 0.9),
        (1.0, 1, 0.8),
        (2.0, 0, 0.95),
    ]
    labels = ["A", "B"]
    out_file = tmp_path / "plot.png"
    pose_visualization.plot_predictions(preds, labels, save_path=str(out_file), show=False)
    assert out_file.exists()
