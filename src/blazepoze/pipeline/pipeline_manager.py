import logging
from typing import Any, Callable

import cv2 as cv

try:  # DepthAI may not be installed when running unit tests
    import depthai as dai
except Exception:  # pragma: no cover - imported at runtime on device
    dai = None  # type: ignore

logger = logging.getLogger(__name__)


class PipelineManager:
    """Utility class to manage DepthAI device lifecycle."""

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline

    @staticmethod
    def validate_device_available() -> None:
        """Ensure that at least one DepthAI device is connected."""
        if dai is None:
            return

        if not dai.Device.getAllAvailableDevices():
            raise RuntimeError(
                "No DepthAI device found! Please ensure that the camera is connected"
            )

    def run(
        self,
        queue_setup: Callable[[Any], None],
        loop_body: Callable[[], None],
    ) -> None:
        """Create the device, initialize queues and execute the loop."""
        if dai is None:
            raise RuntimeError("DepthAI library is not available")

        self.validate_device_available()

        with dai.Device(self.pipeline) as device:
            queue_setup(device)
            loop_body()

        cv.destroyAllWindows()
