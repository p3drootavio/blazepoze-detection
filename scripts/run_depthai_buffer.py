import argparse
import os
# The new class name is PoseActionClassifier
from src.blazepoze.pipeline.depthai_buffer import PoseActionClassifier
from src.blazepoze.visualization.visualization_utils import VisualizationUtils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_labels(label_file: str | None) -> list[str]:
    """Read class labels from ``label_file`` relative to project root."""
    if not label_file:
        return []
    label_path = os.path.join(PROJECT_ROOT, label_file)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(args):
    # Construct full paths to all models
    # The first argument is now the Keras model, not the blob
    keras_model = os.path.join(PROJECT_ROOT, args.keras_model)
    pd_model = os.path.join(PROJECT_ROOT, args.pd_model)
    lm_model = os.path.join(PROJECT_ROOT, args.lm_model)
    # Check if all model files exist
    # Make sure to check the keras_model path
    for model_path in [keras_model, pd_model, lm_model]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    labels = read_labels(args.label_file)

    # Instantiate the classifier with the new Keras model path
    pipeline = PoseActionClassifier(
        keras_model_path=keras_model,
        pd_model_path=pd_model,
        lm_model_path=lm_model,
        labels=labels
    )
    VisualizationUtils.display_enabled = not args.no_display
    pipeline.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pose-based Action Recognition Pipeline")

    # --- Model Arguments ---
    # Change the first argument to point to your .keras file
    parser.add_argument(
        "--keras_model",
        default="models/pretrained/pose_tcn_augmented.keras",
        help="Path to the TCN classifier .keras model.",
    )
    parser.add_argument(
        "--pd_model",
        default="depthai_blazepose/models/pose_detection_sh4.blob",
        help="Path to the pose detection model blob.",
    )
    parser.add_argument(
        "--lm_model",
        default="depthai_blazepose/models/pose_landmark_full_sh4.blob",
        help="Path to the landmark model blob ('full', 'lite', or 'heavy').",
    )
    parser.add_argument(
        "--label_file",
        default="labels.txt",
        help="Optional path to a text file with class labels.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV image display",
    )

    args = parser.parse_args()
    main(args)
