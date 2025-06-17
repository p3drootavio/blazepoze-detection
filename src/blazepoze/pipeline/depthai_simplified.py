"""Simplified DepthAI classifier pipeline.

This module defines :class:`DepthAIClassifier` which implements a single-stage
pipeline for running an image classifier on OAK devices. Frames from the RGB
camera preview are resized on device and sent directly to the classifier
network. Only two output streams are exposed to the host: the high resolution
video feed and the neural network results.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

import cv2 as cv
import numpy as np

try:  # DepthAI may not be installed when running unit tests
    import depthai as dai
except Exception:  # pragma: no cover - imported at runtime on device
    dai = None  # type: ignore


logger = logging.getLogger(__name__)


class DepthAIClassifier:
    """Single-stage classifier pipeline for OAK cameras."""

    def __init__(self, blob_file_path: str, labels: Optional[Iterable[str]] = None) -> None:
        if not os.path.exists(blob_file_path):
            raise FileNotFoundError(f"Classifier blob not found: {blob_file_path}")

        self.blob_file_path = blob_file_path
        self.labels: List[str] = list(labels) if labels else []
        self.pipeline = dai.Pipeline() if dai else None

        # Default values which may get updated by ``_describe_blob``
        self.input_size = (224, 224)
        self.output_layer = ""  # type: str

        self._describe_blob(self.blob_file_path)
        if self.pipeline is not None:
            self._create_pipeline()

    # ------------------------------------------------------------------
    # Pipeline creation helpers
    # ------------------------------------------------------------------
    def _describe_blob(self, blob_path: str) -> None:
        """Print information about the provided blob.

        Attempts to read the blob using DepthAI's OpenVINO utilities. If the
        blob cannot be parsed (for example when running unit tests without the
        DepthAI library), the method falls back to default values.
        """

        if dai is None:
            logger.warning("DepthAI not available; skipping blob description")
            return

        try:
            blob = dai.OpenVINO.Blob(blob_path)
        except Exception as exc:  # pragma: no cover - depends on DepthAI
            logger.warning("Failed to describe blob: %s", exc)
            return

        print("Inputs:")
        for name, info in blob.networkInputs.items():
            print(f" - Name: '{name}', Type: {info.precision}, Shape: {info.dims}")
            # OpenVINO stores dims as NCHW
            if len(info.dims) >= 4:
                self.input_size = (info.dims[3], info.dims[2])

        print("Outputs:")
        for name, info in blob.networkOutputs.items():
            print(f" - Name: '{name}', Type: {info.precision}, Shape: {info.dims}")
            self.output_layer = name

    def _create_pipeline(self) -> None:
        """Configure DepthAI nodes for the single-stage pipeline."""

        assert self.pipeline is not None  # for type checkers

        # Camera node
        self.cam = self.pipeline.create(dai.node.ColorCamera)
        self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.cam.setFps(30)

        # High-resolution video stream
        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        self.cam.video.link(xout_rgb.input)

        # On-device preprocessing
        self.manip = self.pipeline.create(dai.node.ImageManip)
        self.manip.initialConfig.setResize(*self.input_size)
        self.manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        self.cam.preview.link(self.manip.inputImage)

        # Neural network
        self.nn = self.pipeline.create(dai.node.NeuralNetwork)
        self.nn.setBlobPath(self.blob_file_path)
        self.manip.out.link(self.nn.input)

        # NN output to host
        xout_nn = self.pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        self.nn.out.link(xout_nn.input)

    # ------------------------------------------------------------------
    # Device interaction
    # ------------------------------------------------------------------
    def connect_device(self) -> None:
        """Start the pipeline on the connected OAK device."""

        if dai is None:
            raise RuntimeError("DepthAI library is not available")

        self._validate_device_available()

        with dai.Device(self.pipeline) as device:
            rgb_q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
            nn_q = device.getOutputQueue("nn", maxSize=4, blocking=False)

            while True:
                frame = rgb_q.get().getCvFrame()

                if nn_q.has():
                    logits = np.array(nn_q.get().getFirstLayerFp16())
                    label_idx = int(np.argmax(logits))
                    label = self.labels[label_idx] if label_idx < len(self.labels) else str(label_idx)
                    cv.putText(
                        frame,
                        label,
                        (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                cv.imshow("frame", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break

        cv.destroyAllWindows()

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_device_available() -> None:
        if dai is None:  # pragma: no cover - used only when DepthAI is installed
            return

        if not dai.Device.getAllAvailableDevices():
            raise RuntimeError(
                "No DepthAI device found! Please ensure the camera is connected"
            )

