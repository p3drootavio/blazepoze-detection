# Third-party libraries
import argparse
import os

# local modules
from src.blazepoze.pipeline.tnc_model import build_tcn_for_oak as tcnoak
from src.blazepoze.pipeline.pose_dataset import PoseDatasetPipeline
from tensorflow.keras.callbacks import EarlyStopping


def main():
    parser = argparse.ArgumentParser(description="Train OAK-ready model")
    parser.add_argument("--data-dir", default="/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/data", help="Dataset directory")
    parser.add_argument("--output-dir", default="models", help="Directory for outputs")
    args = parser.parse_args()

    SEQUENCE_LENGTH = 50
    LANDMARKS_DIM = 99
    BATCH_SIZE = 32
    DIR_ROOT = args.output_dir

    # Pipeline Initialization
    augmentation_config = {
        "noise": [False, 0.3],
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
    real_shape = (SEQUENCE_LENGTH, LANDMARKS_DIM)
    fake_shape = (3, 10, 165)
    dilations = [1, 2, 4, 8]

    # Create Model and Compile it
    model = tcnoak(
        input_shape_fake=fake_shape,
        real_shape=real_shape,
        filters=64,
        kernel_size=3,
        dilations=dilations,
        num_blocks=4,
        base_rate=0.2,
        output_units=pipeline.class_names_encoded
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train and Test Model
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=[early_stopping]
    )

    # Save Model
    os.makedirs(DIR_ROOT, exist_ok=True)
    model.save(os.path.join(DIR_ROOT, "pose_classifier_oak.keras"))

if __name__ == "__main__":
    main()
