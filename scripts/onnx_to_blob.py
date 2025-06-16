import argparse
from pathlib import Path
import blobconverter as bc


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to DepthAI blob")
    parser.add_argument("onnx_model", help="Path to the ONNX model file")
    parser.add_argument("--shaves", type=int, default=6, help="Number of SHAVES to use")
    parser.add_argument("--output-dir", default="models/deployed", help="Directory to write the blob")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob_path = bc.from_onnx(args.onnx_model, shaves=args.shaves, data_type="FP16", output_dir=str(out_dir))
    print("Saved .blob to:", blob_path)


if __name__ == "__main__":
    main()
