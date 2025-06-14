import os
import depthai as dai
import cv2 as cv
import numpy as np

class DepthAIPipeline:
    """A class to manage and control an OAK (OpenCV AI Kit) camera pipeline.

    This class initializes and manages a DepthAI pipeline for capturing RGB
    frames with a color camera. The setup is easily extendable for additional
    cameras or neural network nodes in the future.
    """
    def __init__(self, blob_file_path, blazepose_blob_path="blazepose.blob"):
        """Initialize the DepthAI pipeline with a color camera configuration.

        Sets up the basic pipeline structure including:
        - Color camera for RGB frames
        - XLink output for the camera stream
        """
        self.blob_file_path = blob_file_path
        self.blazepose_blob_path = blazepose_blob_path
        self.pipeline = dai.Pipeline()

        if not os.path.exists(self.blazepose_blob_path):
            raise FileNotFoundError(
                f"BlazePose blob not found: {self.blazepose_blob_path}"
            )

        if not os.path.exists(self.blob_file_path):
            raise FileNotFoundError(
                f"Classifier blob not found: {self.blob_file_path}"
            )

        self.colorCam = self._getColorCamera(self.pipeline)
        self.xoutRgb = self._createXLinkOut(self.pipeline)

        self.blazepose_out = None

        self.colorCam.video.link(self.xoutRgb.input)


    def __str__(self):
        rgb_configured = hasattr(self, 'colorCam')
        pipeline_created = hasattr(self, 'pipeline')
        nn_configured = hasattr(self, 'nn')
        input_shape = getattr(self, 'fake_input', None)
        input_shape_str = input_shape.shape if input_shape is not None else "Unknown"

        return (
            f"DepthAIPipeline Summary:\n"
            f"  - Blob File Path        : {self.blob_file_path}\n"
            f"  - BlazePose Blob Path   : {self.blazepose_blob_path}\n"
            f"  - Pipeline Created      : {'Yes' if pipeline_created else 'No'}\n"
            f"  - RGB Camera           : {'Configured' if rgb_configured else 'Not Configured'}\n"
            f"  - Resolution           : 1080p\n"
            f"  - Frame Rate           : 30 FPS\n"
            f"  - Neural Network Config : {'Yes' if nn_configured else 'No'}\n"
            f"  - NN Input Shape        : {input_shape_str}\n"
            f"  - NN Input Stream       : {'nn_input' if hasattr(self, 'nn_in') else 'Not Configured'}\n"
            f"  - NN Output Stream      : {'nn_output' if hasattr(self, 'nn_out') else 'Not Configured'}\n"
        )


    def connectDevice(self):
        """Connect to the OAK device and start the video stream."""
        self._validate_device_available()
        self._createNeuralNetworkNodes(self.pipeline)

        try:
            with dai.Device(self.pipeline) as device:
                self._initialize_queues(device)
                self._start_streaming_loop()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            cv.destroyAllWindows()


    def _validate_device_available(self):
        available_devices = dai.Device.getAllAvailableDevices()
        if not available_devices:
            raise RuntimeError("No DepthAI device found! Please ensure that:\n"
                               "1. The OAK camera is properly connected\n"
                               "2. You have necessary permissions to access the device\n"
                               "3. The device is not being used by another application")


    def _initialize_queues(self, device):
        self.rgbQueue = device.getOutputQueue(name="rgb")
        self.blazepose_queue = device.getOutputQueue(name="blazepose_output")
        self.nn_input_queue = device.getInputQueue(name="nn_input")
        self.nn_output_queue = device.getOutputQueue(name="nn_output", maxSize=1, blocking=False)


    def _start_streaming_loop(self):
        cv.namedWindow("window", cv.WINDOW_NORMAL)

        while True:
            frame = self._getFrame(self.rgbQueue)

            cv.imshow("window", frame)

            self._checkKeyboardInput()

            self._prepare_input(self.nn_input_queue)
            self._handle_nn_output()


    def _handle_nn_output(self):
        if self.nn_output_queue.has():
            result = self.nn_output_queue.get()
            prediction = np.array(result.getFirstLayerFp16())
            label = np.argmax(prediction)
            print("Prediction:", label)


    def _createXLinkOut(self, pipeline):
        """Create an XLinkOut node for the pipeline.

        Args:
            pipeline (dai.Pipeline): The DepthAI pipeline instance.

        Returns:
            dai.node.XLinkOut: Configured XLinkOut node for RGB output.
        """
        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        return xout


    def _getColorCamera(self, pipeline):
        """Configure and create a color camera node.

        Sets up an RGB camera with 1080p resolution at 30 FPS.

        Args:
            pipeline (dai.Pipeline): The DepthAI pipeline instance.

        Returns:
            dai.node.ColorCamera: Configured color camera node.
        """
        color = pipeline.create(dai.node.ColorCamera)
        color.setBoardSocket(dai.CameraBoardSocket.RGB)
        color.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color.setInterleaved(False)
        color.setFps(30)
        return color


    def _getFrame(self, queue):
        """Retrieve and convert a frame from the camera queue.

        Args:
            queue (dai.DataOutputQueue): Queue containing camera frames.

        Returns:
            numpy.ndarray: Frame converted to OpenCV format.
        """
        self.frame = queue.get()
        return self.frame.getCvFrame()


    def _checkKeyboardInput(self):
        """Handle keyboard input for camera view control.

        Processes keyboard input for:
        - 'q': Quit the application

        Raises:
            StopIteration: If 'q' is pressed to quit the application.
        """
        key = cv.waitKey(1)
        if key == ord('q'):
            raise StopIteration


    def _createNeuralNetworkNodes(self, pipeline):
        # BlazePose NN (stage 1)
        self.blazepose_nn = pipeline.create(dai.node.NeuralNetwork)
        self.blazepose_nn.setBlobPath(self.blazepose_blob_path)
        self.blazepose_nn.setNumInferenceThreads(2)
        self.blazepose_nn.input.setBlocking(False)
        self.colorCam.video.link(self.blazepose_nn.input)

        # Output from BlazePose
        self.blazepose_out = pipeline.create(dai.node.XLinkOut)
        self.blazepose_out.setStreamName("blazepose_output")
        self.blazepose_nn.out.link(self.blazepose_out.input)

        # Gesture Classifier NN (stage 2)
        self.nn = pipeline.create(dai.node.NeuralNetwork)
        self.nn.setBlobPath(self.blob_file_path)
        self.nn.setNumInferenceThreads(2)
        self.nn.input.setBlocking(False)

        # XLinkIn for manually feeding keypoints
        self.nn_in = pipeline.create(dai.node.XLinkIn)
        self.nn_in.setStreamName("nn_input")
        self.nn_in.out.link(self.nn.input)

        # XLinkOut for gesture output
        self.nn_out = pipeline.create(dai.node.XLinkOut)
        self.nn_out.setStreamName("nn_output")
        self.nn.out.link(self.nn_out.input)


    def _prepare_input(self, input_queue):
        if self.blazepose_queue.has():
            result = self.blazepose_queue.get()
            keypoints = np.array(result.getFirstLayerFp16())  # shape: (4950,) for (1, 3, 10, 165)

            # Optional: validate or reshape if needed
            if keypoints.size != 4950:
                print("Unexpected BlazePose output size:", keypoints.shape)
                return

            nn_data = dai.NNData()
            nn_data.setLayer("oak_input", keypoints)
            input_queue.send(nn_data)

            # Save for __str__
            self.fake_input = keypoints.reshape(1, 3, 10, 165)
