import argparse
import os
from src.blazepoze.pipeline.depthai_simplified import DepthAIClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_labels(label_file):
    if not label_file:
        return []
    label_path = os.path.join(PROJECT_ROOT, label_file)
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found at: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(classifier_blob, label_file=None):
    classifier_blob = os.path.join(PROJECT_ROOT, classifier_blob)
    if not os.path.exists(classifier_blob):
        raise FileNotFoundError(f"Classifier blob not found at: {classifier_blob}")
    labels = read_labels(label_file)
    pipeline = DepthAIClassifier(classifier_blob, labels)
    print("Before connecting the device:", pipeline)
    pipeline.connect_device()
    print("After connecting the device:", pipeline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simplified DepthAI classifier pipeline")
    parser.add_argument(
        "--classifier-blob",
        default="models/deployed/blazepose.blob",
        help="Path to the classifier blob",
    )
    parser.add_argument(
        "--label-file",
        default=None,
        help="Optional path to a text file with class labels",
    )
    args = parser.parse_args()
    main(args.classifier_blob, args.label_file)
