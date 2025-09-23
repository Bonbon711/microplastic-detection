import argparse, os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from model import DualBranchSwinCNNClassifier
from metrics import generate_cam, overlay_cam_on_image

def cam_to_bbox(cam_map, img_size, percentile=85):
    Hc, Wc = cam_map.shape
    W, H = img_size
    t = np.percentile(cam_map, percentile)
    mask = (cam_map >= t).astype(np.uint8)
    if mask.sum() == 0: return (0,0,W-1,H-1)
    ys, xs = np.where(mask > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    x1 = int(round(x1 * W / Wc)); x2 = int(round(x2 * W / Wc))
    y1 = int(round(y1 * H / Hc)); y2 = int(round(y2 * H / Hc))
    return (x1, y1, x2, y2)

def draw_bbox(img, box, label=None):
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline="red", width=3)
    if label: d.text((box[0]+5, box[1]+5), label, fill="red")
    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--output_dir', default='results')
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--class_names', nargs='*', default=['algae','microplastics'])
    p.add_argument('--cam', action='store_true')
    p.add_argument('--bbox', action='store_true')
    p.add_argument('--bbox_percentile', type=int, default=85)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # model + flexible head width from checkpoint
    model = DualBranchSwinCNNClassifier(num_classes=args.num_classes, pretrained=False)
    state = torch.load(args.weights, map_location="cpu")
    fc_keys = [k for k in state.keys() if k.endswith("fc.weight")]
    if fc_keys:
        ckpt_in = state[fc_keys[0]].shape[1]
        if getattr(model.fc, "in_features", None) != ckpt_in:
            import torch.nn as nn
            model.fc = nn.Linear(ckpt_in, args.num_classes)
    model.load_state_dict(state, strict=False)
    model.eval()

    # gather images
    if os.path.isdir(args.source):
        paths = [os.path.join(args.source, f) for f in os.listdir(args.source)
                 if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))]
    else:
        paths = [args.source]

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    for pth in paths:
        img = Image.open(pth).convert('RGB')
        x = tfm(img)
        # one-time safety: adapt head to real fused width if needed
        with torch.no_grad():
            c = model.cnn(x.unsqueeze(0))
            s = model._swin_feats(x.unsqueeze(0))
            F = torch.cat([c,s],1).shape[1]
            if model.fc.in_features != F:
                import torch.nn as nn
                model.fc = nn.Linear(F, args.num_classes)

        logits = model(x.unsqueeze(0))
        pred = int(logits.argmax(1))
        name = args.class_names[pred] if pred < len(args.class_names) else str(pred)
        print(f"{os.path.basename(pth)} -> Predicted class: {name}")

        base = os.path.splitext(os.path.basename(pth))[0]
        if args.cam or args.bbox:
            cam = generate_cam(model, x, class_idx=pred)

        if args.cam:
            overlay = overlay_cam_on_image(img.resize((224,224)), cam)
            overlay.save(os.path.join(args.output_dir, f"{base}_cam.png"))

        if args.bbox:
            box = cam_to_bbox(cam, img.size, percentile=args.bbox_percentile)
            boxed = draw_bbox(img, box, label=name)
            boxed.save(os.path.join(args.output_dir, f"{base}_bbox.png"))

if __name__ == "__main__":
    main()
