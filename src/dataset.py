import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _tfm():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_dataloaders(train_dir, val_dir, batch_size=32):
    tfm = _tfm()
    tr = datasets.ImageFolder(train_dir, transform=tfm)
    va = datasets.ImageFolder(val_dir,   transform=tfm)
    print(f"[data] train={len(tr)} | val={len(va)} | classes={tr.classes}")
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            tr.classes)

def get_dataloaders_auto(root_dir, batch_size=32, val_ratio=0.2, seed=42):
    tfm = _tfm()
    full = datasets.ImageFolder(root_dir, transform=tfm)
    n_val = max(1, int(len(full)*val_ratio))
    n_trn = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(full, [n_trn, n_val], generator=g)
    print(f"[data:auto] root={root_dir} | train={len(tr)} | val={len(va)} | classes={full.classes}")
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            full.classes)
