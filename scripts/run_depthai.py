import argparse
import os
from src.blazepoze.pipeline.depthai import DepthAIPipeline


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(classifier_blob=None, pose_blob=None):
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
        default="models/deployed/blazepose.blob",
        help="Path to the gesture classifier blob",
    )
    parser.add_argument(
        "--pose-blob",
        default="models/deployed/blazepose.blob",
        help="Path to the BlazePose blob",
    )
    args = parser.parse_args()
    main(args.classifier_blob, args.pose_blob)
