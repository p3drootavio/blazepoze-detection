from src.blazepoze.pipeline.depthai import DepthAIPipeline


def main():
    try:
        pipeline = DepthAIPipeline()
        pipeline.connectDevice()
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
