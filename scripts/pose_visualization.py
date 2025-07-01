from __future__ import annotations

"""Visualize pose classification results from :mod:`run_depthai_buffer`.

This script runs the :class:`~src.blazepoze.pipeline.depthai_buffer.PoseActionClassifier`
pipeline and records the predicted class for each processed buffer. After the
run completes a matplotlib plot is generated showing how the predicted
confidence evolves over time for each class.
"""

import argparse
import os
import time
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.blazepoze.pipeline.depthai_buffer import PoseActionClassifier
from src.blazepoze.visualization.visualization_utils import VisualizationUtils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_labels(label_file: str | None) -> List[str]:
    """Read class labels from ``label_file`` relative to project root."""
    if not label_file:
        return []
    label_path = os.path.join(PROJECT_ROOT, label_file)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


Prediction = Tuple[float, int, float]


def run_and_collect(
    classifier: PoseActionClassifier,
    threshold: float = 0.95,
    max_frames: int | None = None,
) -> List[Prediction]:
    """Run the classification pipeline and collect predictions.

    Parameters
    ----------
    classifier : PoseActionClassifier
        Initialized classifier pipeline.
    threshold : float, optional
        Confidence threshold to accept a prediction. Defaults to ``0.95``.
    max_frames : int, optional
        Optional limit on processed frames for early termination.

    Returns
    -------
    list of tuple
        Each tuple is ``(time_sec, label_idx, confidence)``.
    """
    preds: List[Prediction] = []
    start_time = time.perf_counter()
    frames_seen = 0

    current_label = "Waiting for buffer..."
    label_color = (0, 0, 0)

    while True:
        frame, body = classifier.tracker.next_frame()
        if frame is None:
            break

        if body:
            landmarks = body.landmarks_world.flatten()
            classifier.landmark_buffer.append(landmarks)

            if len(classifier.landmark_buffer) == classifier.landmark_buffer.maxlen:
                input_tensor = classifier._prepare_input_tensor()
                scores = classifier.tcn_model.predict(input_tensor)[0]
                label_idx = int(np.argmax(scores))
                confidence = float(np.max(scores))
                elapsed = time.perf_counter() - start_time
                preds.append((elapsed, label_idx, confidence))

                if confidence > threshold:
                    if label_idx < len(classifier.labels):
                        label = classifier.labels[label_idx]
                    else:
                        label = str(label_idx)
                    current_label = f"{label} ({confidence:.2f})"
                    label_color = (0, 255, 0)
                else:
                    current_label = "No pose detected."
                    label_color = (0, 0, 255)
        else:
            if len(classifier.landmark_buffer) < classifier.landmark_buffer.maxlen:
                current_label = (
                    f"Buffer: {len(classifier.landmark_buffer)}/"
                    f"{classifier.landmark_buffer.maxlen}"
                )

        VisualizationUtils.overlay_text(frame, current_label, color=label_color)
        if not VisualizationUtils.show_image("Pose Action Recognition", frame):
            break

        frames_seen += 1
        if max_frames and frames_seen >= max_frames:
            break

    classifier.tracker.exit()
    return preds


def plot_predictions(
    predictions: Sequence[Prediction],
    labels: Iterable[str],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot predictions over time.

    Parameters
    ----------
    predictions : sequence of tuple
        Output from :func:`run_and_collect`.
    labels : iterable of str
        Class labels used to annotate the plot.
    save_path : str, optional
        If provided, the plot is saved to this path.
    show : bool, optional
        Whether to display the plot with ``plt.show()``.
    """
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize pose classification")
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
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV image display",
    )
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


def main() -> None:
    args = build_arg_parser().parse_args()

    keras_model = os.path.join(PROJECT_ROOT, args.keras_model)
    pd_model = os.path.join(PROJECT_ROOT, args.pd_model)
    lm_model = os.path.join(PROJECT_ROOT, args.lm_model)

    for path in [keras_model, pd_model, lm_model]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at: {path}")

    labels = read_labels(args.label_file)

    classifier = PoseActionClassifier(
        keras_model_path=keras_model,
        pd_model_path=pd_model,
        lm_model_path=lm_model,
        labels=labels,
    )

    VisualizationUtils.display_enabled = not args.no_display
    preds = run_and_collect(
        classifier,
        threshold=args.confidence_threshold,
        max_frames=args.max_frames,
    )
    plot_predictions(preds, labels or [str(i) for i in range( len(set(p[1] for p in preds)) )], save_path=args.output_plot)


if __name__ == "__main__":
    main()
