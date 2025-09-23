# src/optimize.py
import argparse
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from model import DualBranchSwinTinyClassifier

def quantize_and_save(weights_path, output_path, num_classes=2):
    # Initialize model without freeze_backbone (not supported)
    model = DualBranchSwinTinyClassifier(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Quantize only Linear layers
    model_quantized = quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # Save quantized model
    torch.save(model_quantized.state_dict(), output_path)
    print(f"Quantized model saved at {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help="Path to trained weights")
    parser.add_argument('--output', type=str, default="quantized_model.pth", help="Output path for quantized model")
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    quantize_and_save(args.weights, args.output, num_classes=args.num_classes)

if __name__ == "__main__":
    main()
