"""
Capture 55 consecutive frames of 3D pose landmarks (33 keypoints × 3 coordinates=99 values) at ~FPS (≈5.5s)
Save them to a CSV file with shape (55 rows × 99 cols)

Usage (example) ───────────────────────────────────────────────────────────────────────────────────────────
source .venv/bin/activate
export PYTHONPATH="$PYTHONPATH:/Users/username/project/path"
python scripts/capture_data_points.py \
  --pd_model depthai_blazepose/models/pose_detection_sh4.blob \
  --lm_model depthai_blazepose/models/pose_landmark_full_sh4.blob \
  --output_dir output
"""

from __future__ import annotations

import argparse
import cv2
import datetime as _dt
import logging
import numpy as np
import pandas as pd
from collections import deque
from pathlib import Path
from typing import Deque

from depthai_blazepose.BlazeposeDepthai import BlazeposeDepthai

# Logging setup
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataPointCapturer:
    """Capture one (buffer_size, 99) landmark tensor and dump it to CSV."""

    #: Expected size of one flattened landmark vector (33keypoints × 3 coords)
    LANDMARK_VECTOR_LEN: int = 99

    def __init__(self, *, pd_model_path: str, lm_model_path: str, output_dir: str, buffer_size: int = 55, fps: int = 10, smoothing: bool = True,) -> None:
        self.buffer_size = buffer_size
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialise BlazePose tracker (runs directly on the OAK‑D device)
        self.tracker = BlazeposeDepthai(
            pd_model=pd_model_path,
            lm_model=lm_model_path,
            smoothing=smoothing,
            xyz=True,
            internal_fps=fps,
        )

        # Circular buffer collecting consecutive landmark vectors
        self.landmark_buffer: Deque[np.ndarray] = deque(maxlen=self.buffer_size)
        logger.info("Tracker initialised. Ready for capture …")


    def capture_once(self) -> np.ndarray:
        """Grab buffer_size landmark frames and return as an array."""
        logger.info(
            "Capturing %d frames (expected duration ≈ %.1fs)…",
            self.buffer_size,
            self.buffer_size / self.fps,
        )

        self.landmark_buffer.clear()
        while len(self.landmark_buffer) < self.buffer_size:
            frame, body = self.tracker.next_frame()
            if frame is None:
                raise RuntimeError("Frame retrieval failed – camera disconnected?")

            if body:  # A person was detected in the frame.
                landmarks = body.landmarks_world.flatten()

                # Guard against occasional malformed outputs
                if landmarks.shape[0] != self.LANDMARK_VECTOR_LEN:
                    logger.debug("Skipping malformed landmark of length %d", landmarks.shape[0])
                    continue

                self.landmark_buffer.append(landmarks)

            # Keep roughly constant sampling rate irrespective of processing time
            cv2.waitKey(int(1000 / self.fps))  # non‑blocking millisecond sleep

        logger.info("Capture finished.")
        return np.array(self.landmark_buffer, dtype=np.float32)


    def save_capture(self, *, file_name: str | None = None) -> Path:
        """Run one capture session and write the CSV. Returns the file path."""
        data = self.capture_once()

        if file_name is None:
            file_name = _dt.datetime.now().strftime("capture_%Y%m%d_%H%M%S")

        csv_path = self.output_dir / f"{file_name}.csv"
        # Write with no header/index – model training code usually expects raw numbers
        pd.DataFrame(data).to_csv(csv_path, header=False, index=False)
        logger.info("Saved CSV → %s", csv_path)
        return csv_path


    def close(self) -> None:
        """Release DepthAI resources and close any OpenCV windows."""
        self.tracker.exit()
        cv2.destroyAllWindows()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Capture 55×99 pose landmark snapshots to CSV")
    p.add_argument("--pd_model", required=True, help="Path to pose detection .blob")
    p.add_argument("--lm_model", required=True, help="Path to pose landmark .blob")
    p.add_argument("--output_dir", default="captures", help="Directory for resulting CSV files")
    p.add_argument("--buffer_size", type=int, default=55, help="Number of consecutive frames (rows)")
    p.add_argument("--fps", type=int, default=10, help="Internal camera FPS (controls capture duration)")
    p.add_argument("--no_smoothing", action="store_true", help="Disable landmark smoothing filter")
    p.add_argument("--file_name", default=None, help="Optional custom CSV base name (no extension)")
    return p


def visualize_points():
    pass



def _main() -> None:
    args = _build_arg_parser().parse_args()

    capturer = DataPointCapturer(
        pd_model_path=args.pd_model,
        lm_model_path=args.lm_model,
        output_dir=args.output_dir,
        buffer_size=args.buffer_size,
        fps=args.fps,
        smoothing=not args.no_smoothing,
    )

    try:
        capturer.save_capture(file_name=args.file_name)
    except KeyboardInterrupt:
        logger.info("Interrupted by user – exiting…")
    finally:
        capturer.close()


if __name__ == "__main__":
    _main()
