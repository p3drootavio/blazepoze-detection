import depthai as dai
import cv2 as cv
import numpy as np

class DepthAIPipeline:
    """A class to manage and control an OAK (OpenCV AI Kit) stereo camera pipeline.

    This class initializes and manages a DepthAI pipeline for stereo vision,
    configuring both left and right mono cameras and their respective output streams.
    """
    def __init__(self):
        """Initialize the DepthAI pipeline with stereo camera configuration.

        Sets up the basic pipeline structure including:
        - Left and right mono cameras
        - XLink outputs for both cameras
        - Links between cameras and their respective outputs
        """
        self.pipeline = dai.Pipeline()

        self.monoLeft = self._getMonoCamera(self.pipeline, isLeft=True)
        self.monoRight = self._getMonoCamera(self.pipeline, isLeft=False)

        self.xoutLeft = self._createXLinkOut(self.pipeline, isLeft=True)
        self.xoutRight = self._createXLinkOut(self.pipeline, isLeft=False)

        self.monoLeft.out.link(self.xoutLeft.input)
        self.monoRight.out.link(self.xoutRight.input)


    def connectDevice(self, sideBySide=True):
        """Connect to the OAK device and start the video stream.

        Creates a connection to the OAK device, initializes output queues,
        and displays the camera feeds in a window. The display can be either
        side-by-side or overlapped views of both cameras.

        Args:
            sideBySide (bool, optional): If True, displays left and right frames
                side by side. If False, displays an overlapped view. Defaults to True.

        Raises:
            RuntimeError: If no DepthAI device is found or cannot be accessed.
            dai.error.DeviceUnavailableError: If device disconnects or is being used
                by another application.
        """
        available_devices = dai.Device.getAllAvailableDevices()
        if len(available_devices) == 0:
            raise RuntimeError("No DepthAI device found! Please ensure that:\n"
                             "1. The OAK camera is properly connected\n"
                             "2. You have necessary permissions to access the device\n"
                             "3. The device is not being used by another application")

        try:
            with dai.Device(self.pipeline) as device:
                self.leftQueue = device.getOutputQueue(name="left")
                self.rightQueue = device.getOutputQueue(name="right")

                # Set display window name
                cv.namedWindow("window", cv.WINDOW_NORMAL)

                while True:
                    leftFrame = self._getFrame(self.leftQueue)
                    rightFrame = self._getFrame(self.rightQueue)

                    if sideBySide:
                        imOut = np.hstack((leftFrame, rightFrame))
                    else:
                        imOut = np.uint8(((leftFrame / 2) + (rightFrame / 2)))

                    cv.imshow("window", imOut)

                    self._checkKeyboardInput(sideBySide)
        except dai.error.DeviceUnavailableError:
            print("Device disconnected or is being used by another application")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            cv.destroyAllWindows()


    def _createXLinkOut(self, pipeline, isLeft=False):
        """Create an XLinkOut node for the pipeline.

        Args:
            pipeline (dai.Pipeline): The DepthAI pipeline instance.
            isLeft (bool, optional): If True, creates output for left camera.
                If False, creates output for right camera. Defaults to False.

        Returns:
            dai.node.XLinkOut: Configured XLinkOut node.
        """
        mono = pipeline.createXLinkOut()
        mono.setStreamName("left") if isLeft else mono.setStreamName("right")
        return mono


    def _getMonoCamera(self, pipeline, isLeft=False):
        """Configure and create a mono camera node.

        Sets up a mono camera with 720p resolution at 40 FPS.

        Args:
            pipeline (dai.Pipeline): The DepthAI pipeline instance.
            isLeft (bool, optional): If True, configures left camera.
                If False, configures right camera. Defaults to False.

        Returns:
            dai.node.MonoCamera: Configured mono camera node.
        """
        # Configure mono camera
        mono = pipeline.createMonoCamera()

        # Set camera resolution
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

        # Set camera FPS
        mono.setFps(40)

        # Set camera name
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT) if isLeft else mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        return mono


    def _getFrame(self, queue):
        """Retrieve and convert a frame from the camera queue.

        Args:
            queue (dai.DataOutputQueue): Queue containing camera frames.

        Returns:
            numpy.ndarray: Frame converted to OpenCV format.
        """
        self.frame = queue.get()
        return self.frame.getCvFrame()


    def _checkKeyboardInput(self, sideBySide):
        """Handle keyboard input for camera view control.

        Processes keyboard input for:
        - 'q': Quit the application
        - 't': Toggle between side-by-side and overlapped view

        Args:
            sideBySide (bool): Current state of the display mode.

        Returns:
            bool: Updated state of sideBySide if 't' is pressed.

        Raises:
            StopIteration: If 'q' is pressed to quit the application.
        """
        key = cv.waitKey(1)
        if key == ord('q'):
            raise StopIteration
        elif key == ord('t'):
            sideBySide = not sideBySide
            return sideBySide
