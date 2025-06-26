# In your ONNX conversion script...
from __future__ import annotations
import tensorflow as tf
import tf2onnx

# Make sure to import the builder functions and the custom layer
from src.blazepoze.pipeline.tnc_model import TemporalBlock, build_tcn_for_oak


class ModelConverter:
    """Convert Keras models to ONNX using tf2onnx.

    Parameters
    ----------
    custom_objects : dict, optional
        Mapping of custom layer names to classes required to load the model.
    """

    def __init__(self, custom_objects: dict | None = None) -> None:
        self.custom_objects = custom_objects or {"TemporalBlock": TemporalBlock}
        self.model: tf.keras.Model | None = None

    def load(self, model_path: str) -> tf.keras.Model:
        """Load a Keras model from ``model_path``."""

        self.model = tf.keras.models.load_model(
            model_path, custom_objects=self.custom_objects
        )
        return self.model

    def convert(
        self,
        model_path: str,
        output_path: str,
        input_shape: tuple[int | None, ...] = (None, 50, 99),
        opset: int = 13,
    ) -> None:
        """Convert a model to ONNX.

        Parameters
        ----------
        model_path : str
            Path to the Keras ``.keras`` model file.
        output_path : str
            Where the ``.onnx`` file will be written.
        input_shape : tuple of ints, optional
            Shape of the input tensor excluding the batch dimension. Defaults to
            ``(None, 50, 99)``.
        opset : int, optional
            ONNX opset version. Defaults to ``13``.
        """

        model = self.load(model_path)
        spec = (
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input"),
        )
        tf2onnx.convert.from_keras(
            model,
            input_signature=spec,
            opset=opset,
            output_path=output_path,
        )


def main() -> None:
    ROOT_DIR = "/models"

    # 1. Define model parameters (ensure these match your trained model)
    #    You may need to load a config file or define them here.
    output_units = 14  # Example: number of your classes
    filters = 32
    kernel_size = 3
    num_blocks = 4
    dilations = [1, 2, 4, 8]

    # These are the key shapes for the OAK-compatible wrapper
    # The total elements must match: 3 * 10 * 165 = 4950 = 50 * 99
    input_shape_fake = (3, 10, 165)
    real_shape = (50, 99)

    # 2. Build the OAK-compatible model in memory
    print("Building OAK-compatible TCN model...")
    oak_model = build_tcn_for_oak(
        input_shape_fake=input_shape_fake,
        real_shape=real_shape,
        filters=filters,
        kernel_size=kernel_size,
        dilations=dilations,
        num_blocks=num_blocks,
        output_units=output_units
    )
    # It's good practice to load the weights from your trained model
    # This assumes your 'pose_tcn_new.keras' was built with the standard `build_tcn`
    print("Loading weights from trained model...")
    oak_model.load_weights(f"{ROOT_DIR}/pretrained/pose_tcn_new.keras")

    # 3. Define the input signature for the OAK model
    #    The shape must now match the 'fake' input of the wrapper
    spec = (
        tf.TensorSpec(shape=(None, *input_shape_fake), dtype=tf.float32, name="input"),
    )

    output_path = f"{ROOT_DIR}/exported/pose_tcn_new_oak.onnx"

    # 4. Convert the new model object to ONNX
    print(f"Converting model to ONNX at {output_path}...")
    tf2onnx.convert.from_keras(
        oak_model,
        input_signature=spec,
        opset=13,  # Using opset 13 is fine
        output_path=output_path,
    )
    print("Conversion successful!")


if __name__ == "__main__":
    main()