from src.pipeline import object_pipeline

# Pipeline Initialization
augmentation_config = {
    "noise": [True, 0.3],
    "scale": [True, 1.0],
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


# Visualize a sample with augmentation applied
pipeline.plot_sample(augment=False, save_fig=True)


# Train datasets for training, validation, and testing
train_ds = pipeline.get_tf_dataset("train", augment=True)
val_ds = pipeline.get_tf_dataset("val")
test_ds = pipeline.get_tf_dataset("test")

# Get information about the pipeline
print(pipeline)


'''
# Save pipeline and configurations
pipeline.save_pipeline("/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/output",
                       save_config=True)

# Load pipeline and configurations
pipeline.load_pipeline("/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/output",
                       True,
                       "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/config/pipeline.config.json")
'''