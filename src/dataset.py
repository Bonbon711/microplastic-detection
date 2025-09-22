import os, json
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

from model import IMAGENET_MEAN, IMAGENET_STD

IM_SIZE = 224

def _read_rgb(path: Path):
    """
    Robust image loader that supports Windows paths with non-ASCII characters.
    Uses np.fromfile + cv2.imdecode instead of cv2.imread.
    """
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _clahe_rgb(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    lab = cv2.merge([cl, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def _center_resize(img, size=IM_SIZE):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_t = (size - nh) // 2
    pad_b = size - nh - pad_t
    pad_l = (size - nw) // 2
    pad_r = size - nw - pad_l
    return cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r,
                              borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

def _to_tensor_norm(x_f32):
    x = torch.from_numpy(x_f32).permute(2,0,1)
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(IMAGENET_STD).view(3,1,1)
    return (x - mean) / std

_a_train = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.RandomResizedCrop(IM_SIZE, IM_SIZE, scale=(0.7,1.0), ratio=(0.9,1.1)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(var_limit=(5.0,20.0), p=0.2),
    A.MotionBlur(p=0.1),
])

_a_val = A.Compose([
    A.SmallestMaxSize(max_size=IM_SIZE),
    A.CenterCrop(IM_SIZE, IM_SIZE),
])

class DualViewFolder(Dataset):
    """
    root/
      algae/*.jpg|png
      microplastics/*.jpg|png
    """
    def __init__(self, root_dir, split="train"):
        root = Path(root_dir)
        classes = [("algae",0), ("microplastics",1)]
        self.items = []
        for cname, label in classes:
            cdir = root / cname
            if not cdir.exists():
                raise FileNotFoundError(f"Missing folder: {cdir}")
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                    self.items.append((cdir / fn, label))
        self.split = split

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = _read_rgb(path)
        img_cla = _clahe_rgb(img)

        if self.split == "train":
            img = _a_train(image=img)["image"]
            img_cla = _a_train(image=img_cla)["image"]
        else:
            img = _a_val(image=_center_resize(img))["image"]
            img_cla = _a_val(image=_center_resize(img_cla))["image"]

        x_std = _to_tensor_norm((img.astype(np.float32))/255.0)
        x_cla = _to_tensor_norm((img_cla.astype(np.float32))/255.0)
        y = torch.tensor(label, dtype=torch.long)
        return x_std, x_cla, y
