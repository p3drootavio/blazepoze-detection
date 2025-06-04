# Convert ONNX Model to .blob for OAK-D Deployment (Non-Image Model)

This guide explains how to convert a non-image ONNX model (e.g., a Temporal Convolutional Network for pose classification) into a `.blob` file compatible with Luxonis OAK-D cameras.

---

## Prerequisites

- Python 3.8–3.10
- Git, CMake, and a C++ compiler (e.g., `build-essential` on Linux)
- Your model saved in ONNX format: `tcn_model.onnx`
- A Windows or Linux machine with access to OpenVINO (macOS not supported for blob conversion)

---

## Step 1: Install OpenVINO

### Linux

```bash
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/openvino-2023.0.0.tar.gz
tar -xf openvino-2023.0.0.tar.gz
cd openvino-2023.0.0
source setupvars.sh
```

### Windows

Download and install from:
[https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html)

Then run in Command Prompt:
```cmd
"C:\Program Files (x86)\Intel\openvino_2023\setupvars.bat"
```

---

## Step 2: Convert ONNX → OpenVINO IR

```bash
mo \
  --input_model /path/to/pose_tcn.onnx \
  --input_shape "[1,50,99]" \
  --data_type FP16 \
  --output_dir /path/to/output_ir
```

This creates:
- `tcn_model.xml`
- `tcn_model.bin`

---

## Step 3: Compile IR → .blob Using Luxonis `compile_tool`

### Clone and Build the Tool

```bash
git clone https://github.com/luxonis/compile_tool.git
cd compile_tool
cmake .
make
```

### Run Compilation

```bash
./compile_tool \
  -m /path/to/output_ir/tcn_model.xml \
  -o /path/to/blob_output \
  -sh 6
```

This will produce:
```
tcn_model_openvino_2022.1_6shaves.blob
```

---

## Step 4: Transfer `.blob` to Deployment Device (e.g., macOS + OAK-D)

You can now move the `.blob` file to your deployment machine using USB, SCP, or cloud sync.

---

## Notes

- This process is required because `blobconverter` cloud API only supports image-based models.
- The `compile_tool` approach supports **custom data** like time-series or landmarks.
- You must preprocess the input data to match your ONNX model (e.g., shape `[1, 50, 99]`) before sending it to the OAK-D pipeline.

---
