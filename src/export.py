from pathlib import Path
import torch
import torch.nn as nn

from model import DualBranchSwinTinyClassifier
from dataset import IM_SIZE

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "classifier.pth"

def export():
    device = torch.device("cpu")
    model = DualBranchSwinTinyClassifier(num_classes=2, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # dummy dual inputs (std & clahe), CHW
    x1 = torch.randn(1,3,IM_SIZE,IM_SIZE)
    x2 = torch.randn(1,3,IM_SIZE,IM_SIZE)

    # TorchScript (trace)
    traced = torch.jit.trace(model, (x1, x2))
    ts_path = ROOT / "microplastic_classifier_scripted.pt"
    traced.save(str(ts_path))
    print(f"TorchScript (quantized, traced) saved: {ts_path}")

    # ONNX (float)
    onnx_path = ROOT / "microplastic_classifier.onnx"
    torch.onnx.export(
        model, (x1, x2), str(onnx_path),
        input_names=["x_std","x_clahe"], output_names=["logits"],
        opset_version=12, dynamic_axes={"x_std":{0:"batch"}, "x_clahe":{0:"batch"}, "logits":{0:"batch"}}
    )
    print(f"ONNX (float) saved: {onnx_path}")

if __name__ == "__main__":
    export()
