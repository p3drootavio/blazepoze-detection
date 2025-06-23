import argparse
import os
import yaml
from src.blazepoze.pipeline.depthai import DepthAIPipeline
from src.blazepoze.visualization.visualization_utils import VisualizationUtils


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

def main(classifier_blob=None, pose_blob=None, resolution="1080p", fps=30, confidence_threshold=0.5):
    try:
        # Resolve to full paths relative to project root
        classifier_blob = os.path.join(PROJECT_ROOT, classifier_blob)
        pose_blob = os.path.join(PROJECT_ROOT, pose_blob)

        if not os.path.exists(classifier_blob):
            raise FileNotFoundError(f"Classifier blob not found at: {classifier_blob}")

        if not os.path.exists(pose_blob):
            raise FileNotFoundError(f"BlazePose blob not found at: {pose_blob}")

        pipeline = DepthAIPipeline(
            blob_file_path=classifier_blob,
            blazepose_blob_path=pose_blob,
            resolution=resolution,
            fps=fps,
            confidence_threshold=confidence_threshold,
        )
        print("Before connecting the device: ", pipeline)
        pipeline.connectDevice()
        print("After connecting the device: ", pipeline)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DepthAI gesture pipeline")
    parser.add_argument(
        "--classifier-blob",
        default=CONFIG.get("depthai", {}).get("classifier_blob", "models/deployed/pose_classifier_oak_openvino_2022.1_6shave.blob"),
        help="Path to the gesture classifier blob",
    )
    parser.add_argument(
        "--pose-blob",
        default=CONFIG.get("depthai", {}).get("pose_blob", "depthai_blazepose/models/pose_landmark_full_sh4.blob"),
        help="Path to the BlazePose blob",
    )
    parser.add_argument(
        "--resolution",
        default=CONFIG.get("depthai", {}).get("camera_resolution", "1080p"),
        help="Camera resolution (1080p, 720p, 4k)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=CONFIG.get("depthai", {}).get("fps", 30),
        help="Camera frame rate",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIG.get("depthai", {}).get("confidence_threshold", 0.5),
        help="Minimum confidence to display prediction",
    )
    args = parser.parse_args()
    main(args.classifier_blob, args.pose_blob, args.resolution, args.fps, args.threshold)