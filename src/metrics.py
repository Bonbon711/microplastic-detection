import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import matplotlib.pyplot as plt
import pandas as pd  # for Excel export


# ---------- CAM utilities (used by detect.py too) ----------
def generate_cam(model, image_tensor, class_idx=None):
    """
    Grad-CAM on the CNN branch's last conv layer (ResNet18 layer4).
    image_tensor: 3D (C,H,W) normalized as in training.
    Returns a [0,1] heatmap (h, w).
    """
    model.eval()
    feature_maps, gradients = {}, {}

    def forward_hook(module, inp, out):
        feature_maps['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # NOTE: register_full_backward_hook would silence a deprecation warning,
    # but layer4 supports this standard hook fine for our use.
    hook_f = model.cnn.layer4.register_forward_hook(forward_hook)
    hook_b = model.cnn.layer4.register_backward_hook(backward_hook)

    outputs = model(image_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = int(outputs.argmax(dim=1))
    model.zero_grad()
    target = outputs[0, class_idx]
    target.backward()

    fmap = feature_maps['value'][0]   # [C, h, w]
    grads = gradients['value'][0]     # [C, h, w]
    hook_f.remove(); hook_b.remove()

    weights = grads.mean(dim=(1, 2))  # [C]
    cam_map = torch.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        cam_map += w * fmap[i]
    cam_map = torch.relu(cam_map)
    cam_map = cam_map - cam_map.min()
    cam_map = cam_map / (cam_map.max() + 1e-8)
    return cam_map.cpu().numpy()


def overlay_cam_on_image(img: Image.Image, cam_map: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    H, W = img.size[1], img.size[0]
    heatmap = (cm.jet(cam_map) * 255).astype(np.uint8)[:, :, :3]
    heatmap = Image.fromarray(heatmap).resize((W, H), resample=Image.BILINEAR)
    heatmap = np.array(heatmap)
    img_np  = np.array(img.convert('RGB'))
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------- Full evaluation over a folder ----------
@torch.no_grad()
def evaluate_folder(weights, data_dir, out_dir="results", class_names=None, batch_size=32, num_classes=2):
    from model import DualBranchSwinCNNClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchSwinCNNClassifier(num_classes=num_classes, pretrained=False).to(device)

    # --- Robustly adapt classifier head to the checkpoint's input width ---
    state = torch.load(weights, map_location=device)
    fc_keys = [k for k in state.keys() if k.endswith("fc.weight")]
    if fc_keys:
        ckpt_in = state[fc_keys[0]].shape[1]
        if getattr(model.fc, "in_features", None) != ckpt_in:
            import torch.nn as nn
            model.fc = nn.Linear(ckpt_in, num_classes).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    # Dataset / Loader
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(str(data_dir), transform=tfm)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    # One-time safety: adapt head to actual fused width from this dataset
    if len(ds) > 0:
        probe_x, _ = next(iter(dl))
        probe_x = probe_x.to(device)
        feat_cnn  = model.cnn(probe_x)                 # [B, C1]
        feat_swin = model._swin_feats(probe_x)         # [B, C2]
        fused     = torch.cat([feat_cnn, feat_swin], 1)  # [B, F]
        F = fused.shape[1]
        if model.fc.in_features != F:
            import torch.nn as nn
            model.fc = nn.Linear(F, num_classes).to(device)

    # Track predictions + file paths
    y_true, y_pred = [], []
    paths_all = [p for p, _ in ds.samples]
    path_cursor = 0
    per_image_rows = []

    for x, y in dl:
        bs = x.size(0)
        batch_paths = paths_all[path_cursor:path_cursor+bs]
        path_cursor += bs

        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy().tolist())

        # per-image log
        for pth, t, pr in zip(batch_paths, y.numpy().tolist(), preds):
            per_image_rows.append({
                "filename": Path(pth).name,
                "true_class": class_names[t] if class_names else str(t),
                "predicted_class": class_names[pr] if class_names else str(pr),
                "correct": (t == pr)
            })

    # Metrics
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=range(num_classes),
        target_names=class_names if class_names else None,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(outp / "metrics_report.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "precision_per_class": prec.tolist(),
            "recall_per_class": rec.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "classification_report": report
        }, f, indent=2)

    # Confusion matrix PNG
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    ticks = class_names if class_names else list(range(num_classes))
    plt.xticks(range(num_classes), ticks, rotation=45)
    plt.yticks(range(num_classes), ticks)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outp / "confusion_matrix.png")
    plt.close(fig)

    # ---- Excel export (one file, three sheets) ----
    per_image_df = pd.DataFrame(per_image_rows)
    per_class_df = pd.DataFrame({
        "class": class_names if class_names else list(range(num_classes)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support
    })
    summary_df = pd.DataFrame([{
        "accuracy": acc,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "samples": int(report["weighted avg"]["support"])
    }])

    xlsx_path = outp / "metrics_report.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            per_image_df.to_excel(writer, index=False, sheet_name="per_image")
            per_class_df.to_excel(writer, index=False, sheet_name="per_class")
            summary_df.to_excel(writer, index=False, sheet_name="summary")
        print(f"Saved Excel metrics → {xlsx_path}")
    except Exception as e:
        # Fallback: also save CSVs if Excel engine is missing
        per_image_df.to_csv(outp / "per_image.csv", index=False)
        per_class_df.to_csv(outp / "per_class.csv", index=False)
        summary_df.to_csv(outp / "summary.csv", index=False)
        print(f"Excel write failed ({e}). Wrote CSVs instead to {outp}")

    # Console summary
    print(f"Accuracy: {acc:.4f}")
    print("Saved metrics to results/metrics_report.json, metrics_report.xlsx (or CSVs), and confusion_matrix.png")
    return acc, cm, report


def main():
    ap = argparse.ArgumentParser("Compute accuracy/F1/recall/precision + confusion matrix for a folder")
    ap.add_argument("--data", required=True, help="Folder with class subfolders (e.g., algae/ microplastics/)")
    ap.add_argument("--weights", required=True, help="Path to classifier .pth")
    ap.add_argument("--out", default="results", help="Where to save metrics outputs")
    ap.add_argument("--class_names", nargs="*", default=["algae","microplastics"], help="Class names in order")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_classes", type=int, default=2)
    args = ap.parse_args()

    evaluate_folder(args.weights, args.data, out_dir=args.out,
                    class_names=args.class_names,
                    batch_size=args.batch_size,
                    num_classes=args.num_classes)

if __name__ == "__main__":
    main()
