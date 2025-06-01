import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pipeline import object_pipeline
from src.pipeline import tnc_model
from sklearn.metrics import confusion_matrix, classification_report

# Pipeline Initialization
augmentation_config = {
    "noise": [False, 0.3],
    "scale": [False, 1.0],
    "shift": [False, 0.0]
}

pipeline = object_pipeline.PoseDatasetPipeline(
    data_dir="/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/data",
    sequence_length=50,
    landmarks_dim=99,
    batch_size=32,
    augmentation_config=augmentation_config
)

# Load and Split Data
pipeline.load_data()
pipeline.split_data()

# Visualize Data
pipeline.plot_landmarks_distribution(sample_size=500)
pipeline.plot_landmarks_across_time(joint_index=32)
pipeline.plot_landmarks_clustered()

# Create Datasets
train_dataset = pipeline.get_tf_dataset("train", augment=False)
val_dataset = pipeline.get_tf_dataset("val")
test_dataset = pipeline.get_tf_dataset("test")

input_shape = (50, 99)
dilations = [1, 2, 4, 8, 16, 32]

# Create the model
model = tnc_model.build_tcn(
    input_shape=input_shape,
    filters=64,
    kernel_size=3,
    dilations=dilations,
    num_blocks=6,
    dropout_rate=0.2,
    output_units=pipeline.get_classes_encoder()
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train and test the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    batch_size=32,
)

loss = model.evaluate(test_dataset)
print(f"Test Loss: {loss}", end="\n\n")

X_batch, y_true = None, None
for batch in test_dataset.take(1):
    X_batch, y_true = batch

y_pred = model.predict(X_batch)
predicted_classes = np.argmax(y_pred, axis=1)

class_names = pipeline.get_classes()
predicted_labels = [class_names[i] for i in predicted_classes]
true_labels = [class_names[i] for i in y_true.numpy()]

print(f"Predicted labels: {predicted_labels}", end="\n\n")
print(f"True labels: {true_labels}", end="\n\n")

report = classification_report(y_true, predicted_classes, target_names=class_names, output_dict=True)
print(report)

report_df = pd.DataFrame(report).transpose()

plt.matshow(report_df.drop('support', axis=1))
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=90)
plt.yticks(range(len(class_names)), class_names)
plt.title("Classification Report")
plt.show()
