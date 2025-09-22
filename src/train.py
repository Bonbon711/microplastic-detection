# src/train.py
import argparse
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

# ------- try to import your dataset, fallback to ImageFolder -------
try:
    from dataset import MicroplasticsDualInputDataset  # preferred
    HAS_CUSTOM_DS = True
except Exception:
    HAS_CUSTOM_DS = False
    from torchvision import datasets, transforms

# ------- resolve a model class from model.py dynamically -------
def resolve_model_class():
    import importlib
    m = importlib.import_module("model")
    candidate_names = [
        "DualBranchSwinTinyClassifier",   # if you still have it
        "TinyStudentCNN",                 # lightweight student we used
        "MobileNetV3SmallClassifier",     # other light options
        "EfficientNetLiteClassifier",
        "SimpleCNN",
    ]
    for name in candidate_names:
        if hasattr(m, name):
            return getattr(m, name), name
    # as a last resort, look for any nn.Module subclass
    for attr in dir(m):
        obj = getattr(m, attr)
        if isinstance(obj, type) and issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module:
            return obj, attr
    raise ImportError("No usable model class found in model.py")

# ------- reproducibility -------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------- make loaders (works with custom DS or ImageFolder) -------
def make_loaders(data_dir: Path, batch_size: int = 16, val_ratio: float = 0.2, num_workers: int = 2):
    if HAS_CUSTOM_DS:
        full = MicroplasticsDualInputDataset(str(data_dir))
        # expect __getitem__ -> (x_std, x_clahe, y)
        labels = [int(full[i][2]) for i in range(len(full))]
        indices = np.arange(len(full))
    else:
        # Fallback using ImageFolder (expects data_dir/algae and data_dir/microplastics)
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalize to ImageNet stats (okay for most pretrained backbones)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        full_if = datasets.ImageFolder(str(data_dir), transform=tfm)
        full = full_if
        labels = [full.targets[i] for i in range(len(full))]
        indices = np.arange(len(full))

    # stratified split
    labels_np = np.array(labels)
    n_total = len(indices)
    n_val = int(round(val_ratio * n_total))
    # simple stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_val, random_state=42)
    train_idx, val_idx = next(sss.split(indices, labels_np))

    if HAS_CUSTOM_DS:
        train_ds = Subset(full, train_idx.tolist())
        val_ds = Subset(full, val_idx.tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        train_ds = Subset(full, train_idx.tolist())
        val_ds   = Subset(full, val_idx.tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # class counts on training set
    binc = np.bincount(labels_np[train_idx], minlength=2)
    return train_loader, val_loader, (len(train_idx), len(val_idx)), binc

# ------- class weights for imbalance -------
def class_weights_from_counts(counts):
    # inverse frequency
    counts = counts.astype(np.float32)
    counts[counts == 0] = 1.0
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32)

# ------- training / evaluation -------
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad()
        if HAS_CUSTOM_DS:
            x_std, x_clahe, y = batch
            x_std, x_clahe, y = x_std.to(device), x_clahe.to(device), y.to(device)
            logits = model(x_std, x_clahe)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    losses, correct, total = [], 0, 0
    for batch in loader:
        if HAS_CUSTOM_DS:
            x_std, x_clahe, y = batch
            x_std, x_clahe, y = x_std.to(device), x_clahe.to(device), y.to(device)
            logits = model(x_std, x_clahe)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    acc = correct / max(total, 1)
    return (float(np.mean(losses)) if losses else 0.0), acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to preprocessed dataset (two folders: algae/ microplastics/)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_model", type=str, default="classifier.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data)
    assert data_dir.exists(), f"Data path not found: {data_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_loader, val_loader, (n_tr, n_val), binc = make_loaders(data_dir, batch_size=args.batch_size)
    print(f"Train set: {n_tr} images (class counts: {binc.tolist()})")
    print(f"Val set:   {n_val} images")

    # model
    ModelClass, picked = resolve_model_class()
    print(f"Using model: {picked}")
    # most of our lightweight classifiers take num_classes=2 and optionally freeze_backbone
    try:
        model = ModelClass(num_classes=2, freeze_backbone=True)
    except TypeError:
        # fallback to only num_classes
        model = ModelClass(num_classes=2)
    model.to(device)

    # loss (weighted CE for imbalance)
    w = class_weights_from_counts(binc).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    out_path = Path(args.output_model)

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ New best. Saved to {out_path.resolve()}")

    # write a small training summary
    summary = {
        "data": str(data_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_acc": best_acc,
        "model_class": picked,
        "class_counts_train": binc.tolist(),
    }
    (Path("results")).mkdir(exist_ok=True)
    with open(Path("results") / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results/train_summary.json")

if __name__ == "__main__":
    main()
