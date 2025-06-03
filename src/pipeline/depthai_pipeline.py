import depthai as dai
import cv2 as cv
import numpy as np

class DepthAIPipeline:
    def __init__(self):
        self.pipeline = dai.Pipeline()

        self.monoLeft = self._getMonoCamera(self.pipeline, isLeft=True)
        self.monoRight = self._getMonoCamera(self.pipeline, isLeft=False)

        self.xoutLeft = self._createXLinkOut(self.pipeline, isLeft=True)
        self.xoutRight = self._createXLinkOut(self.pipeline, isLeft=False)

        self.monoLeft.out.link(self.xoutLeft.input)
        self.monoRight.out.link(self.xoutRight.input)


    def connectdDvice(self, sideBySide=True):
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
        mono = pipeline.createXLinkOut()
        mono.setStreamName("left") if isLeft else mono.setStreamName("right")
        return mono


    def _getMonoCamera(self, pipeline, isLeft=False):
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
        self.frame = queue.get()
        return self.frame.getCvFrame()


    def _checkKeyboardInput(self, sideBySide):
        key = cv.waitKey(1)
        if key == ord('q'):
            raise StopIteration
        elif key == ord('t'):
            sideBySide = not sideBySide
            return sideBySide
