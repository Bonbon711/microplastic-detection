import argparse
import os
from PIL import Image
import torch
from torchvision import transforms
from model import DualBranchSwinCNNClassifier
from metrics import generate_cam, overlay_cam_on_image

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True, help='Path to trained model weights (.pth)')
parser.add_argument('--source', type=str, required=True, help='Path to an image or directory of images')
parser.add_argument('--cam', action='store_true', help='If set, output CAM heatmap overlay for each image')
parser.add_argument('--output_dir', type=str, default='results', help='Directory to save output images')
parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (must match the trained model)')
args = parser.parse_args()

# Load the model and weights
model = DualBranchSwinCNNClassifier(num_classes=args.num_classes, pretrained=False)
state_dict = torch.load(args.weights, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Create output directory if needed
os.makedirs(args.output_dir, exist_ok=True)

# Determine list of images to process
if os.path.isdir(args.source):
    image_paths = [os.path.join(args.source, f) for f in os.listdir(args.source) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
else:
    image_paths = [args.source]

# Image preprocessing (resize to 224x224 and normalize as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)  # preprocess
    output = model(img_tensor.unsqueeze(0))
    pred_class = int(output.argmax(dim=1))
    print(f"{os.path.basename(img_path)} -> Predicted class: {pred_class}")
    if args.cam:
        # Generate CAM heatmap for the predicted class
        cam_map = generate_cam(model, img_tensor, class_idx=pred_class)
        # Overlay CAM on the original image (resized to model input size for alignment)
        cam_overlay = overlay_cam_on_image(img.resize((224, 224)), cam_map)
        out_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(img_path))[0] + "_cam.png")
        cam_overlay.save(out_path)
        print(f"Saved CAM visualization to {out_path}")
