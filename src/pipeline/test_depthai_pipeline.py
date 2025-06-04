from src.pipeline.depthai_pipeline import DepthAIPipeline


def main():
    try:
        pipeline = DepthAIPipeline()
        pipeline.connectDevice()
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
