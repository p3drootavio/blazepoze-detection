"""
Compute the number of floating-point operations (FLOPs) in a Keras model.

Example
-------
PYTHONPATH=$(pwd)/.. python calculate_flops.py --model-path ../.../.../model.keras
"""

import argparse
import tensorflow as tf
from src.blazepoze.pipeline.tnc_model import TemporalBlock
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def get_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Freeze a Keras model, import it into a plain TF graph, "
                    "and report total FLOPs.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the .keras file (or SavedModel directory).")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Numeric precision used when tracing the graph (default: float32).")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Core logic
# --------------------------------------------------------------------------- #
def main() -> None:
    args = get_cli_args()

    # 1. Load model (register custom layers if any)
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"TemporalBlock": TemporalBlock}
    )

    # 2. Trace with a concrete input signature
    @tf.function
    def model_fn(x):
        return model(x)

    input_shape = [
        1 if (i == 0 or dim is None) else dim     # batch = 1; replace None with 1
        for i, dim in enumerate(model.input_shape)
    ]
    concrete_fn = model_fn.get_concrete_function(
        tf.TensorSpec(input_shape, getattr(tf, args.dtype))
    )

    # 3. Freeze to GraphDef
    frozen = convert_variables_to_constants_v2(concrete_fn)
    graph_def = frozen.graph.as_graph_def()

    # 4. Re-import and profile
    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(graph_def, name="")
        opts = ProfileOptionBuilder.float_operation()
        flops = profile(graph=g, options=opts)

    total_flops = flops.total_float_ops
    print(f"Total FLOPs : {total_flops:,}")
    print(f"Total GFLOPs: {total_flops / 1e9:.2f}")


if __name__ == "__main__":
    main()
