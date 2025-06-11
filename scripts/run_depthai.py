import os
from src.blazepoze.pipeline.depthai import DepthAIPipeline


def main(file=None):
    try:
        if not file:
            raise ValueError("Blob file path not provided")

        if not os.path.exists(file):
            raise FileNotFoundError(f"Blob file not found at path: {file}")

        pipeline = DepthAIPipeline(
            blob_file_path=file
        )
        print("Before connecting the device: ", pipeline)
        pipeline.connectDevice()
        print("After connecting the device: ", pipeline)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    blob_file_path = "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/models/deployed/pose_classifier_oak.blob"
    main(blob_file_path)
