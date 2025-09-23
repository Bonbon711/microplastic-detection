# src/export.py
import argparse
import torch
from model import DualBranchSwinTinyClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help="Path to trained model")
    parser.add_argument('--output', type=str, default="model.onnx", help="ONNX output file")
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    # Model without invalid freeze_backbone arg
    model = DualBranchSwinTinyClassifier(num_classes=args.num_classes, pretrained=False)
    state_dict = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )
    print(f"Exported model to {args.output}")

if __name__ == "__main__":
    main()
