import blobconverter
from pathlib import Path

# Define the root path to your models for clean path management
model_dir = Path("/Users/pedrootavionascimentocamposdeoliveira/PycharmProjects/HIVE/models/deployed")
model_name = "pose_tcn_new"

# Construct the full paths using pathlib
xml_path = model_dir / f"{model_name}.xml"
bin_path = model_dir / f"{model_name}.bin"

try:
    # Convert the Path objects to strings before passing them to the function
    blob_path = blobconverter.from_openvino(
        xml=str(xml_path),
        bin=str(bin_path),
        data_type="FP16",
        shaves=6,
        version="2022.0"
    )
    print(f"Blob saved successfully to: {blob_path}")

except Exception as e:
    print(f"An error occurred during blob conversion: {e}")