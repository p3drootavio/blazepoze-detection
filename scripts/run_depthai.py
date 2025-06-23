import argparse
import os
from src.blazepoze.pipeline.depthai import DepthAIPipeline
from src.blazepoze.visualization.visualization_utils import VisualizationUtils


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(classifier_blob=None, pose_blob=None, display=True):
    VisualizationUtils.display_enabled = display
    try:
        # Resolve to full paths relative to project root
        classifier_blob = os.path.join(PROJECT_ROOT, classifier_blob)
        pose_blob = os.path.join(PROJECT_ROOT, pose_blob)

        if not os.path.exists(classifier_blob):
            raise FileNotFoundError(f"Classifier blob not found at: {classifier_blob}")

        if not os.path.exists(pose_blob):
            raise FileNotFoundError(f"BlazePose blob not found at: {pose_blob}")

        pipeline = DepthAIPipeline(
            blob_file_path=classifier_blob,  # stage-2
            blazepose_blob_path=pose_blob,   # stage-1
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
        default="models/deployed/pose_classifier_oak_openvino_2022.1_6shave.blob",
        help="Path to the gesture classifier blob",
    )
    parser.add_argument(
        "--pose-blob",
        default="/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/depthai_blazepose/models/pose_landmark_full_sh4.blob",
        help="Path to the BlazePose blob",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV image display",
    )
    args = parser.parse_args()
    main(args.classifier_blob, args.pose_blob, not args.no_display)
