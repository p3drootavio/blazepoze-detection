# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping

# local modules
from src.pipeline import tnc_model
from src.pipeline.object_pipeline import PoseDatasetPipeline
from src.visualization.pose_visualizer import PoseVisualizer

def main():
    SEQUENCE_LENGTH = 50
    LANDMARKS_DIM = 99
    BATCH_SIZE = 32
    EPOCHS = 30
    DIR_ROOT = "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch"

    # Pipeline Initialization
    augmentation_config = {
        "noise": [False, 0.3],
        "scale": [False, 1.0],
        "shift": [False, 0.0]
    }

    pipeline = PoseDatasetPipeline(
        data_dir=DIR_ROOT + "/data",
        sequence_length=SEQUENCE_LENGTH,
        landmarks_dim=LANDMARKS_DIM,
        batch_size=BATCH_SIZE,
        augmentation_config=augmentation_config
    )

    # Load and Split Data
    pipeline.load_data()
    pipeline.split_data()

    # Visualize Data
    visualizer = PoseVisualizer(pipeline)
    visualizer.plot_landmarks_distribution(sample_size=500)
    visualizer.plot_landmarks_across_time(joint_index=32)
    visualizer.plot_landmarks_clustered()

    # Create Datasets
    train_dataset = pipeline.get_tf_dataset("train", augment=False)
    val_dataset = pipeline.get_tf_dataset("val")
    test_dataset = pipeline.get_tf_dataset("test")

    # TCN Model Configuration
    input_shape = (50, 99)
    dilations = [1, 2, 4, 8, 16, 32]

    # Create Model and Compile it
    model = tnc_model.build_tcn(
        input_shape=input_shape,
        filters=64,
        kernel_size=3,
        dilations=dilations,
        num_blocks=6,
        dropout_rate=0.2,
        output_units=pipeline.class_names_encoded,
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Train and Test Model
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        batch_size=32,
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
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Save Model
    model.save(DIR_ROOT + "/output/model.keras")
    pd.DataFrame(history.history).to_csv(DIR_ROOT + "/output/history.csv", index=False)

if __name__ == "__main__":
    main()
