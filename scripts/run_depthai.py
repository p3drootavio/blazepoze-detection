import argparse
import os
from src.blazepoze.pipeline.depthai import DepthAIPipeline


def main(classifier_blob=None, pose_blob=None):
    try:
        if not classifier_blob or not pose_blob:
            raise ValueError("Blob file path not provided")

        if not os.path.exists(classifier_blob):
            raise FileNotFoundError(f"Blob file not found at path: {classifier_blob}")

        if not os.path.exists(pose_blob):
            raise FileNotFoundError(f"Blob file not found at path: {pose_blob}")

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
        default="models/deployed/pose_classifier_oak.blob",
        help="Path to the gesture classifier blob",
    )
    parser.add_argument(
        "--pose-blob",
        default="blazepose.blob",
        help="Path to the BlazePose blob",
    )
    args = parser.parse_args()
    main(args.classifier_blob, args.pose_blob)
