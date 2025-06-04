import tensorflow as tf
import tf2onnx

from src.blazepoze.pipeline.tnc_model_strong import TemporalBlock

ROOT_DIR = "/models"
custom_objects = {"TemporalBlock": TemporalBlock}

# Load the Keras model
model = tf.keras.models.load_model(ROOT_DIR + "/pose_tcn.keras", custom_objects=custom_objects)

# Convert the Keras model to ONNX
spec = (tf.TensorSpec(shape=(None, 50, 99), dtype=tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=ROOT_DIR + "/pose_tcn.onnx")
