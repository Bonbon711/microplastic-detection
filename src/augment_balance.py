# src/augment_balance.py
import argparse, random
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def imread(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def aug_once(img: np.ndarray, use_clahe=True) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()

    # --- flips ---
    if random.random() < 0.5:
        out = cv2.flip(out, 1)  # horizontal
    if random.random() < 0.2:
        out = cv2.flip(out, 0)  # vertical (occasionally)

    # --- small rotation + scale ---
    ang = random.uniform(-12, 12)
    scale = random.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, scale)
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # --- brightness / contrast ---
    alpha = random.uniform(0.9, 1.1)    # contrast
    beta  = random.uniform(-12, 12)     # brightness
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # --- slight color jitter (HSV) ---
    if random.random() < 0.6:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-10, 10), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-4, 4)) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- mild blur or noise ---
    if random.random() < 0.4:
        if random.random() < 0.5:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        else:
            noise = np.random.normal(0, 6, out.shape).astype(np.int16)
            out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- CLAHE on L channel (optional, microscopy-friendly) ---
    if use_clahe and random.random() < 0.6:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return out

def count_images(d: Path) -> int:
    return sum(1 for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)

def main():
    ap = argparse.ArgumentParser(description="Augment a single class up to a target count.")
    ap.add_argument("--input", required=True, help="Root dataset folder with class subfolders (e.g., algae/microplastics)")
    ap.add_argument("--outdir", required=True, help="Output root (balanced copy is written here).")
    ap.add_argument("--class", dest="cls", required=True, choices=["algae", "microplastics"], help="Class to augment")
    ap.add_argument("--target", type=int, required=True, help="Desired total images for that class (e.g., 900)")
    ap.add_argument("--max_per_src", type=int, default=20, help="Maximum augmented copies generated per single source image")
    ap.add_argument("--use_clahe", action="store_true", help="Apply CLAHE randomly in augmentation")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.outdir)
    ensure_dir(out_root)

    # Mirror the whole dataset first (copy all originals to outdir)
    for cls_name in ["algae", "microplastics"]:
        src = in_root / cls_name
        dst = out_root / cls_name
        ensure_dir(dst)
        imgs = [p for p in src.glob("*") if p.suffix.lower() in IMG_EXTS]
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[WARN] skip unreadable: {p}")
                continue
            cv2.imwrite(str(dst / p.name), img)

    # Now augment ONLY the requested class to reach target
    cls_dir = out_root / args.cls
    cur = count_images(cls_dir)
    if cur >= args.target:
        print(f"[INFO] {args.cls} already has {cur} >= target {args.target}. Nothing to do.")
        return

    # Load originals from INPUT for augmentation diversity
    source_dir = in_root / args.cls
    sources = [p for p in source_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    if not sources:
        print(f"[ERROR] No images found in {source_dir}")
        return

    print(f"[START] Augmenting '{args.cls}' from {cur} → {args.target}")
    per_src_counts = {p:0 for p in sources}
    idx = 0
    while cur < args.target:
        p = sources[idx % len(sources)]
        if per_src_counts[p] >= args.max_per_src:
            idx += 1
            continue
        try:
            img = imread(p)
        except Exception as e:
            print(f"[WARN] {e}; skipping.")
            idx += 1
            continue

        aug_img = aug_once(img, use_clahe=args.use_clahe)
        stem, ext = p.stem, p.suffix
        out_name = f"{stem}_aug{per_src_counts[p]+1:03d}{ext}"
        cv2.imwrite(str(cls_dir / out_name), aug_img)
        per_src_counts[p] += 1
        cur += 1
        idx += 1
        if cur % 50 == 0:
            print(f"  → {cur}/{args.target}")

    print(f"[DONE] {args.cls}: {cur} images in {cls_dir}")

if __name__ == "__main__":
    main()
