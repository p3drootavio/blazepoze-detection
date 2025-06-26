"""
Modified DepthAI pipeline for Pose-based Action Classification.
...
"""
from __future__ import annotations
import logging
import os
import time
from typing import Iterable, List, Optional
from collections import deque

import cv2
import numpy as np
import depthai as dai
import tensorflow as tf

# Import the custom layer definition so Keras knows what a "TemporalBlock" is
from src.blazepoze.pipeline.tnc_model import TemporalBlock

from depthai_blazepose.BlazeposeDepthai import BlazeposeDepthai

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PoseActionClassifier:
    """
    Pipeline for pose-based action classification.
    ...
    """
    def __init__(
        self,
        # We now need the path to the .keras model, not the .blob
        keras_model_path: str,
        pd_model_path: str,
        lm_model_path: str,
        labels: Optional[Iterable[str]] = None,
        smoothing: bool = True
    ) -> None:
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(f"Keras model not found: {keras_model_path}")

        self.labels: List[str] = list(labels) if labels else []

        # --- Host-Side TCN Model Setup ---
        # Load the TCN model with TensorFlow/Keras to run on the CPU
        logger.info("Loading Keras TCN model for host-side inference...")
        self.tcn_model = tf.keras.models.load_model(keras_model_path, compile=False)
        self.input_shape = self.tcn_model.input_shape[1:]
        logger.info(f"Keras model loaded. Expects input shape: {self.input_shape}")


        # --- Pose Estimation Setup (remains the same) ---
        self.tracker = BlazeposeDepthai(
            pd_model=pd_model_path,
            lm_model=lm_model_path,
            smoothing=smoothing,
            xyz=True,
            internal_fps=10,
        )

        # --- Buffer Setup (remains the same) ---
        self.landmark_buffer = deque(maxlen=50)  # Use model's expected length (50)

    def _prepare_input_tensor(self) -> np.ndarray:
        """
        Takes the landmark buffer, converts it to the required tensor shape
        for the Keras TCN model.
        """
        # 1. Create the (50, 99) array from the 50 frames of landmarks in the buffer
        landmark_array = np.array(list(self.landmark_buffer), dtype=np.float32)

        # 2. Reshape to the model's required "fake" input shape (3, 10, 165)
        #    The total number of elements (4950) remains the same.
        #reshaped_array = landmark_array.reshape(3, 10, 165)

        # 3. Add the batch dimension for a final shape of (1, 3, 10, 165)
        #    This is what the model's predict() function expects.
        return np.expand_dims(landmark_array, axis=0)


    def run(self) -> None:
        """
        Starts the pipeline and runs the main loop.
        """
        # We no longer need to validate the device here, as BlazePose does it internally
        # self._validate_device_available() # <-- REMOqVE OR COMMENT OUT

        # We no longer manage a device or pipeline here
        # with dai.Device(self.pipeline) as device: # <-- REMOVE

        current_label = "Waiting for buffer..."
        label_color = (0, 0, 0)

        while True:
            frame, body = self.tracker.next_frame()
            if frame is None:
                break

            if body:
                landmarks = body.landmarks_world.flatten()
                logging.info(f"Landmark shape: {landmarks.shape}")
                self.landmark_buffer.append(landmarks)

                if len(self.landmark_buffer) == self.landmark_buffer.maxlen:
                    # Buffer is full, prepare for host inference
                    input_tensor = self._prepare_input_tensor()

                    # --- Run TCN inference on the host CPU and measure time ---
                    start_time = time.perf_counter()
                    predictions = self.tcn_model.predict(input_tensor)[0]
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(f"TCN inference time: {duration_ms:.2f} ms")

                    # --- Configure the label and the confidence level ---
                    label_idx = np.argmax(predictions)
                    confidence = np.max(predictions)

                    if confidence > 0.7:
                        label = self.labels[label_idx] if label_idx < len(self.labels) else str(label_idx)
                        current_label = f"{label} ({confidence:.2f})"
                        label_color = (0, 0, 0)
                    else:
                        current_label = f"No pose detected."
                        label_color = (0, 0, 0)

            else: # No body detected
                if len(self.landmark_buffer) < self.landmark_buffer.maxlen:
                    buffer_fill = len(self.landmark_buffer)
                    current_label = f"Buffer: {buffer_fill}/{self.landmark_buffer.maxlen}"

            cv2.putText(
                frame,
                current_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                label_color,
                2,
            )
            cv2.imshow("Pose Action Recognition", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        self.tracker.exit()
        cv2.destroyAllWindows()
