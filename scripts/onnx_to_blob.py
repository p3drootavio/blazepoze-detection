import argparse
from pathlib import Path
import blobconverter as bc


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to DepthAI blob. NOTE: blobconverter is a legacy tool."
    )
    parser.add_argument("onnx_model", help="Path to the ONNX model file")
    parser.add_argument("--shaves", type=int, default=6, help="Number of SHAVEs to use")
    parser.add_argument("--output-dir", default="models/deployed", help="Directory to write the blob")
    parser.add_argument("--openvino-version", type=str, help="Specify the OpenVINO version (e.g., 2022.1)")
    parser.add_argument("--data-type", type=str, default="FP16", help="Data type for conversion (e.g., FP16)")
    parser.add_argument("--optimizer-params", nargs='*', help="Additional Model Optimizer parameters")

    args = parser.parse_args()

    print("WARNING: blobconverter is a legacy tool and is no longer actively maintained.")
    print("Consider using Luxonis HubAI for model conversion for better support and performance.")

    onnx_path = Path(args.onnx_model)
    if not onnx_path.is_file():
        print(f"Error: Input ONNX file not found at: {onnx_path}")
        return

    out_dir = Path(args.output_dir)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        print("Starting model conversion...")
        blob_path = bc.from_onnx(
            model=str(onnx_path),
            output_dir=str(out_dir),
            data_type=args.data_type,
            shaves=args.shaves,
            version=args.openvino_version,
            optimizer_params=args.optimizer_params
        )
        print("Successfully saved .blob to:", blob_path)

    except Exception as e:
        print(f"\nAn error occurred during model conversion: {e}")


if __name__ == "__main__":
    main()