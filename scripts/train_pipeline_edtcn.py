# Third-party libraries
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.src.layers import GlobalAveragePooling1D
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# local modules
from src.blazepoze.pipeline.tcn_model_ed import build_ed_tcn as tcn
from src.blazepoze.pipeline.pose_dataset import PoseDatasetPipeline


def main():
    parser = argparse.ArgumentParser(description="Train ED-TCN model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    SEQUENCE_LENGTH = 50
    LANDMARKS_DIM = 99
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DIR_ROOT = args.output_dir

    # Pipeline Initialization
    augmentation_config = {
        "noise": [True, 0.3],
        "scale": [False, 1.0],
        "shift": [False, 0.0]
    }

    pipeline = PoseDatasetPipeline(
        data_dir=args.data_dir,
        sequence_length=SEQUENCE_LENGTH,
        landmarks_dim=LANDMARKS_DIM,
        batch_size=BATCH_SIZE,
        augmentation_config=augmentation_config
    )

    # Load and Split Data
    pipeline.load_data()
    pipeline.split_data()

    # Create Datasets
    train_dataset = pipeline.get_tf_dataset("train", augment=False)
    val_dataset = pipeline.get_tf_dataset("val")
    test_dataset = pipeline.get_tf_dataset("test")

    # Ensure datasets are not empty
    if train_dataset is None or val_dataset is None or test_dataset is None:
        raise ValueError("One or more datasets are empty")

    # TCN Model Configuration
    input_shape = (SEQUENCE_LENGTH, LANDMARKS_DIM)
    num_classes = pipeline.class_names_encoded

    # Create Model and Compile it
    model = tcn(
        input_shape=input_shape,
        filters=64,
        kernel_size=3,
        num_layers=3,
        base_dropout=0.3,
        output_units=num_classes,
        causal=True
    )

    # Add global pooling layer
    output = model.layers[-1].output
    output = GlobalAveragePooling1D()(output)
    model = tf.keras.Model(inputs=model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train and Test Model
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=BATCH_SIZE,
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
    model.save(os.path.join(DIR_ROOT, "pretrained", "pose_tcn.keras"))
    pd.DataFrame(history.history).to_csv(os.path.join(DIR_ROOT, "history.csv"), index=False)


if __name__ == "__main__":
    main()
