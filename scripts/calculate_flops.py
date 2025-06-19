import tensorflow as tf
from src.blazepoze.pipeline.tnc_model_strong import TemporalBlock
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

# 1. Load model with custom layer
model_path = "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/models/pretrained/pose_classifier_oak.keras"
model = tf.keras.models.load_model(model_path,
                                   custom_objects={"TemporalBlock": TemporalBlock})

# 2. Trace once with a concrete input signature
@tf.function
def model_fn(x):
    return model(x)

concrete_fn = model_fn.get_concrete_function(
    tf.TensorSpec([1, 3, 10, 165], tf.float32)   # (batch, C, T, F)
)

# 3. Freeze to a GraphDef
frozen = convert_variables_to_constants_v2(concrete_fn)
graph_def = frozen.graph.as_graph_def()

# 4. Re-import into a plain tf.Graph and profile
with tf.Graph().as_default() as g:
    tf.compat.v1.import_graph_def(graph_def, name="")      # bring ops into `g`
    opts = ProfileOptionBuilder.float_operation()          # count all FP ops
    flops = profile(graph=g, options=opts)

total_flops = flops.total_float_ops
print(f"Total FLOPs: {total_flops:,}")
print(f"Total GFLOPs: {total_flops / 1e9:.2f}")
