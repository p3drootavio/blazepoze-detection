from src.pipeline import object_pipeline

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
pipeline.get_tf_dataset("train", augment=False)
pipeline.get_tf_dataset("val")
pipeline.get_tf_dataset("test")
