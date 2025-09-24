import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
 
def read_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def gray_world_white_balance(img_rgb):
    img = img_rgb.astype(np.float32) + 1e-6
    mean = img.reshape(-1,3).mean(axis=0)
    gain = mean.mean()/mean
    wb = np.clip(img*gain, 0, 255).astype(np.uint8)
    return wb

def illumination_flatten(img_rgb, sigma=15):
    bg = cv2.GaussianBlur(img_rgb, (0,0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.addWeighted(img_rgb, 1.2, bg, -0.5, 128)
    return np.clip(flat, 0, 255).astype(np.uint8)

def clahe_on_v(img_rgb, clip=2.0, tile=8):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile)).apply(v)
    hsv = cv2.merge([h,s,v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def denoise_soft(img_rgb):
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, 3, 3, 7, 21)

def contrast_stretch(img_rgb, low=2, high=98):
    prc1 = np.percentile(img_rgb, low)
    prc2 = np.percentile(img_rgb, high)
    out = np.clip((img_rgb - prc1) * (255.0/(prc2 - prc1 + 1e-6)), 0, 255)
    return out.astype(np.uint8)

def resize_image(img_rgb, size=224):
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

def process_one(img_rgb, size=224):
    x = resize_image(img_rgb, size=size)       # ✅ resize first
    x = gray_world_white_balance(x)
    x = illumination_flatten(x, sigma=31)
    x = clahe_on_v(x, clip=2.0, tile=8)
    x = denoise_soft(x)
    x = contrast_stretch(x, 2, 98)  # safer stretch

    # --- safety check ---
    mean_val = x.mean()
    if mean_val < 5 or mean_val > 250:
        print(f"[AUTO-CORRECT] mean={mean_val:.2f}, adjusting...")

        # Retry with gentler settings
        x = resize_image(img_rgb, size=size)
        x = gray_world_white_balance(x)
        x = illumination_flatten(x, sigma=15)   # gentler flatten
        x = clahe_on_v(x, clip=1.5, tile=8)     # softer CLAHE
        x = denoise_soft(x)
        x = contrast_stretch(x, 5, 95)          # safer contrast

        # If still broken, fall back to original resized
        mean_val2 = x.mean()
        if mean_val2 < 5 or mean_val2 > 250:
            print(f"[FALLBACK] mean={mean_val2:.2f}, using original resized image.")
            x = resize_image(img_rgb, size=size)

    return x

def main():
    ap = argparse.ArgumentParser("Batch preprocessing: resize + illumination flatten + WB + CLAHE + denoise + stretch")
    ap.add_argument("--input", required=True, help="Image folder to read (class subfolders allowed)")
    ap.add_argument("--outdir", default="data_preprocessed", help="Output folder")
    ap.add_argument("--keep-structure", action="store_true", help="Preserve subfolder names (e.g., algae/microplastics)")
    ap.add_argument("--size", type=int, default=224, help="Target size (width=height)")
    args = ap.parse_args()

    in_dir  = Path(args.input)
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.keep_structure:
        subdirs = [d for d in in_dir.iterdir() if d.is_dir()]
        items = []
        for d in subdirs:
            dst = out_dir/d.name
            dst.mkdir(parents=True, exist_ok=True)
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
                    items.append((p, dst/p.name))
    else:
        items = []
        for p in sorted(in_dir.iterdir()):
            if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
                items.append((p, out_dir/p.name))

    ok = 0
    for src, dst in tqdm(items, desc="preprocess", unit="img"):
        try:
            rgb = read_rgb(src)
            out = process_one(rgb, size=args.size)
            cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            ok += 1
        except Exception as e:
            print(f"[WARN] {src.name}: {e}")

    print(f"Done. {ok}/{len(items)} images written to {out_dir}")

if __name__ == "__main__":
    main()
