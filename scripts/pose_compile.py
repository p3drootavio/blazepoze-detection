import blobconverter as bc

blob_path = bc.from_onnx(
    "/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/models/exported/pose_classifier_oak.onnx",
    data_type="FP16",
    shaves=6,
    output_dir="/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/hiveLabResearch/models/deployed"
)
print("Saved .blob to:", blob_path)
print(blob_path)