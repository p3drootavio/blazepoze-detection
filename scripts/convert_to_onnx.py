from __future__ import annotations

import argparse
import tensorflow as tf
import tf2onnx
from pathlib import Path

from src.blazepoze.pipeline.tnc_model import TemporalBlock

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


def build_arg_parser() -> argparse.ArgumentParser:
    prj_root = Path(__file__).resolve().parents[1]  # “…/HIVE”
    default_model = prj_root / "models" / "pretrained" / "pose_tcn.keras"
    default_out   = prj_root / "models" / "converted"  / "pose_tcn.onnx"

    p = argparse.ArgumentParser(description="Convert a Keras model to ONNX")
    p.add_argument(
        "--model-path",
        type=Path,
        default=default_model,
        help=f"Path to .keras model (default: {default_model})",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=default_out,
        help=f"Destination .onnx file (default: {default_out})",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    print(f"[INFO] project root    : {Path(__file__).resolve().parents[1]}")
    print(f"[INFO] model_path      : {args.model_path}")
    print(f"[INFO] output_path     : {args.output_path}")

    converter = ModelConverter()
    converter.convert(
        model_path=args.model_path,
        output_path=args.output_path,
        opset=args.opset,
    )
    print("[SUCCESS] Conversion completed.")


if __name__ == "__main__":
    main()