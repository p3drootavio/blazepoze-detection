# Third-party libraries
import argparse
import os
from pathlib import Path

import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping

# local modules
from src.blazepoze.pipeline.rnn_model import build_rnn as rnn
from src.blazepoze.pipeline.pose_dataset import PoseDatasetPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train pose classification model")
    parser.add_argument(
        "--data-dir",
        default=CONFIG.get("training", {}).get("data_dir", "data"),
        help="Directory with training data",
    )
    parser.add_argument(
        "--output-dir",
        default=CONFIG.get("training", {}).get("output_dir", "models/pretrained"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG.get("training", {}).get("epochs", 30),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG.get("training", {}).get("batch_size", 32),
    )
    args = parser.parse_args()

    SEQUENCE_LENGTH = 50
    LANDMARKS_DIM = 99
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DIR_ROOT = args.output_dir

    # Pipeline Initialization
    augmentation_config = {
        "noise": [True, 0.7],
        "scale": [True, 0.3],
        "shift": [True, 0.2]
    }

    # Resolve data_dir relative to the project root **once**
    data_dir = Path(args.data_dir)

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"Resolved data_dir = {data_dir}")

    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir  # <── key line

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if not any(data_dir.iterdir()):
        raise ValueError(f"Data directory is empty: {data_dir}")

    pipeline = PoseDatasetPipeline(
        data_dir=str(data_dir),
        sequence_length=SEQUENCE_LENGTH,
        landmarks_dim=LANDMARKS_DIM,
        batch_size=BATCH_SIZE,
        augmentation_config=augmentation_config
    )

    # Load and Split Data
    try:
        pipeline.load_data()
        pipeline.split_data()
    except Exception as e:
        raise ValueError(f"Failed to load or split data: {str(e)}")

    # Create Datasets
    train_dataset = pipeline.get_tf_dataset("train", augment=True)
    val_dataset = pipeline.get_tf_dataset("val")
    test_dataset = pipeline.get_tf_dataset("test")

    # Ensure datasets are not empty
    datasets = {
        "training": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }

    empty_datasets = [name for name, ds in datasets.items() if ds is None]
    if empty_datasets:
        raise ValueError(f"The following datasets are empty: {', '.join(empty_datasets)}")

    # TCN Model Configuration
    input_shape = (SEQUENCE_LENGTH, LANDMARKS_DIM)
    output_units = pipeline.class_names_encoded

    # Create Model and Compile it
    model = rnn(input_shape, 64, 128, output_units=output_units)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train and Test Model
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    # Evaluate Model
    loss = model.evaluate(test_dataset)
    print(f"Test Loss: {loss}", end="\n\n")

    # Predict on Test Dataset
    X_batch, y_true = None, None
    for batch in test_dataset.take(1):
        X_batch, y_true = batch

    y_pred = model.predict(X_batch)
    predicted_classes = np.argmax(y_pred, axis=1)

    class_names = pipeline.class_names
    predicted_labels = [class_names[i] for i in predicted_classes]
    true_labels = [class_names[i] for i in y_true.numpy()]

    print(f"Predicted labels: {predicted_labels}", end="\n\n")
    print(f"True labels: {true_labels}", end="\n\n")


    # Classification Report
    report = classification_report(y_true, predicted_classes, target_names=class_names, output_dict=True)
    print(report)

    report_df = pd.DataFrame(report).transpose()

    plt.matshow(report_df.drop('support', axis=1))
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.title("Classification Report")
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Save Model
    os.makedirs(os.path.join(DIR_ROOT, "pretrained"), exist_ok=True)
    model.save(os.path.join(DIR_ROOT, "pretrained", "pose_rnn_augmented.keras"))
    pd.DataFrame(history.history).to_csv(os.path.join(DIR_ROOT, "history.csv"), index=False)


if __name__ == "__main__":
    main()
