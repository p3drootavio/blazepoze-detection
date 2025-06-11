from src.blazepoze.pipeline.depthai import DepthAIPipeline


def main():
    try:
        pipeline = DepthAIPipeline(
            blob_file_path="/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/models/deployed/pose_classifier_oak.blob"
        )
        print("Before connecting the device: ", pipeline)
        pipeline.connectDevice()
        print("After connecting the device: ", pipeline)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
