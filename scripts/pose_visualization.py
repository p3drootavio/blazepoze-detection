"""Visualize pose classification results from :mod:`run_depthai_buffer`.

This script runs the :class:`~src.blazepoze.pipeline.depthai_buffer.PoseActionClassifier`
pipeline and records the predicted class for each processed buffer. After the
run completes a matplotlib plot is generated showing how the predicted
confidence evolves over time for each class.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import helpers without pulling in heavy dependencies from run_depthai_buffer
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_labels(label_file: str | None) -> list[str]:
    """Read class labels from ``label_file`` relative to project root."""
    if not label_file:
        return []
    label_path = os.path.join(PROJECT_ROOT, label_file)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

try:  # Optional heavy imports – may not be available during unit tests
    from src.blazepoze.pipeline.depthai_buffer import PoseActionClassifier
    from src.blazepoze.visualization.visualization_utils import VisualizationUtils
except Exception:  # pragma: no cover - handled in simple test environment
    PoseActionClassifier = None  # type: ignore
    VisualizationUtils = None  # type: ignore

Prediction = Tuple[float, int, float]  # (elapsed_sec, label_idx, confidence)


class PoseActionVisualizer(PoseActionClassifier if PoseActionClassifier else object):
    """Run the TCN pipeline **and** keep a history of predictions.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :class:`PoseActionClassifier`.
    """

    WINDOW_NAME: str = "Pose Action Recognition"


    def __init__(self, *args, **kwargs) -> None:
        if PoseActionClassifier is None:
            raise ImportError("PoseActionClassifier unavailable")
        super().__init__(*args, **kwargs)
        self._start_time: float | None = None
        self.predictions: List[Prediction] = [] # [Tuple(float, int, float), ...]


    def run_and_collect(self, *, threshold: float = 0.9, max_frames: int | None = None) -> List[Prediction]:
        """Execute the tracker loop while storing predictions.

        Parameters
        ----------
        threshold : float, default=0.9
            Minimum confidence required to *accept* a prediction.
        max_frames : int, optional
            Stop after processing this many frames (useful for testing).

        Returns
        -------
        list[Prediction]
            ``(time_sec, label_idx, confidence)`` triples.
        """
        self._start_time = time.perf_counter()
        frames_seen = 0

        current_label = "Waiting for buffer..."
        label_color = (0, 0, 0)  # BGR

        while True:
            frame, body = self.tracker.next_frame()
            if frame is None:
                break  # end‑of‑stream or error → exit main loop

            if body:
                landmarks = body.landmarks_world.flatten()
                self.landmark_buffer.append(landmarks)

                # Only predict once the buffer is full
                if len(self.landmark_buffer) == self.landmark_buffer.maxlen:
                    input_tensor = self._prepare_input_tensor()
                    scores = self.tcn_model.predict(input_tensor)[0]

                    label_idx = int(np.argmax(scores))
                    confidence = float(np.max(scores))
                    elapsed = time.perf_counter() - self._start_time
                    self.predictions.append((elapsed, label_idx, confidence))

                    if confidence > threshold:
                        label = (
                            self.labels[label_idx]
                            if label_idx < len(self.labels)
                            else str(label_idx)
                        )
                        current_label = f"{label} ({confidence:.2f})"
                        label_color = (0, 255, 0)  # green = accepted
                    else:
                        current_label = "No pose detected."
                        label_color = (0, 0, 255)  # red = below threshold
            else:
                # Buffer not full yet → inform progress
                if len(self.landmark_buffer) < self.landmark_buffer.maxlen:
                    current_label = (
                        f"Buffer: {len(self.landmark_buffer)}/"
                        f"{self.landmark_buffer.maxlen}"
                    )

            VisualizationUtils.overlay_text(frame, current_label, color=label_color)
            if not VisualizationUtils.show_image(self.WINDOW_NAME, frame):
                break  # user pressed *Esc* or closed the window

            frames_seen += 1
            if max_frames and frames_seen >= max_frames:
                break

        self.tracker.exit()
        return self.predictions

    # ------------------------------------------------------------------
    # Plotting helpers – kept separate to allow re‑use without rerunning
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        *,
        labels: Sequence[str],
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Render a *confidence‑over‑time* chart for the collected data."""
        if not self.predictions:
            return  # nothing to plot

        times, label_idx, confs = map(np.array, zip(*self.predictions))

        plt.style.use("dark_background")
        plt.figure(figsize=(10, 6))

        for idx, name in enumerate(labels):
            mask = label_idx == idx  # Boolean mask: True where the predicted label matches the current class index
            if np.any(mask):  # skip classes never predicted
                plt.plot(times[mask], confs[mask], marker="o", label=name)

        plt.xlabel("Time (s)")
        plt.ylabel("Confidence")
        plt.title("Pose Classification Over Time")
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)

        if show:
            plt.show()
        plt.close()


def plot_predictions(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot ``(time, label_idx, confidence)`` triples as a confidence timeline."""

    if not predictions:
        return

    times, label_idx, confs = map(np.array, zip(*predictions))

    plt.style.use("dark_background")
    plt.figure(figsize=(10, 6))

    for idx, name in enumerate(labels):
        mask = label_idx == idx
        if np.any(mask):
            plt.plot(times[mask], confs[mask], marker="o", label=name)

    plt.xlabel("Time (s)")
    plt.ylabel("Confidence")
    plt.title("Pose Classification Over Time")
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()
    plt.close()


def _build_arg_parser() -> argparse.ArgumentParser:  # underscore = private helper
    p = argparse.ArgumentParser(description="Visualise pose classification over time")
    p.add_argument(
        "--keras_model",
        default="models/pretrained/pose_tcn_augmented.keras",
        help="Path to the TCN classifier .keras model.",
    )
    p.add_argument(
        "--pd_model",
        default="depthai_blazepose/models/pose_detection_sh4.blob",
        help="Path to the pose detection model blob.",
    )
    p.add_argument(
        "--lm_model",
        default="depthai_blazepose/models/pose_landmark_full_sh4.blob",
        help="Path to the landmark model blob.",
    )
    p.add_argument(
        "--label_file",
        default="labels.txt",
        help="Optional path to a text file with class labels.",
    )
    p.add_argument("--no-display", action="store_true", help="Disable OpenCV image display")
    p.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Minimum confidence to record a prediction.",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit on processed frames.",
    )
    p.add_argument(
        "--output-plot",
        default=None,
        help="Save the generated plot to this path.",
    )
    return p


def main() -> None:  # noqa: D401 – simple script entry‑point
    args = _build_arg_parser().parse_args()

    keras_model = os.path.join(PROJECT_ROOT, args.keras_model)
    pd_model = os.path.join(PROJECT_ROOT, args.pd_model)
    lm_model = os.path.join(PROJECT_ROOT, args.lm_model)

    for path in (keras_model, pd_model, lm_model):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

    labels = read_labels(args.label_file)

    vis = PoseActionVisualizer(
        keras_model_path=keras_model,
        pd_model_path=pd_model,
        lm_model_path=lm_model,
        labels=labels,
    )

    VisualizationUtils.display_enabled = not args.no_display

    vis.run_and_collect(threshold=args.confidence_threshold, max_frames=args.max_frames)

    # Use provided labels if any, otherwise build fallback names based on indices
    vis.plot_predictions(
        labels=labels or [str(i) for i in range(len(set(p[1] for p in vis.predictions)))],
        save_path=args.output_plot,
    )


if __name__ == "__main__":
    main()
