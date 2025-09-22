import torch
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def generate_cam(model, image_tensor, class_idx=None):
    """
    Compute a class activation heatmap (Grad-CAM) for the given image tensor.
    image_tensor: 3D torch Tensor (C,H,W) of a single image (normalized as in training).
    class_idx: class index to generate CAM for (default: model's predicted class).
    Returns: heatmap as a 2D numpy array normalized to [0,1].
    """
    model.eval()
    # Prepare hooks to capture feature map and gradients from last conv layer of CNN branch
    feature_maps = {}
    gradients = {}
    def forward_hook(module, inp, out):
        feature_maps['value'] = out.detach()
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()
    # Register hooks on ResNet18 layer4 (final convolution block output)
    hook_f = model.cnn.layer4.register_forward_hook(forward_hook)
    hook_b = model.cnn.layer4.register_backward_hook(backward_hook)
    # Forward pass
    outputs = model(image_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = int(outputs.argmax(dim=1))  # predicted class index
    # Backward pass for the target class to get gradients
    model.zero_grad()
    target = outputs[0, class_idx]
    target.backward()
    # Get captured data
    fmap = feature_maps['value'][0]   # shape [512, H, W] from ResNet layer4
    grads = gradients['value'][0]     # shape [512, H, W]
    hook_f.remove(); hook_b.remove()  # remove hooks
    # Global average pool the gradients to get weight for each feature channel:contentReference[oaicite:11]{index=11}
    weights = grads.mean(dim=(1, 2))  # shape [512]
    # Compute weighted sum of feature maps
    cam_map = torch.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        cam_map += w * fmap[i]
    cam_map = torch.relu(cam_map)     # only keep positive contributions:contentReference[oaicite:12]{index=12}
    # Normalize heatmap to [0,1]
    cam_map = cam_map - cam_map.min()
    cam_map = cam_map / (cam_map.max() + 1e-8)
    return cam_map.cpu().numpy()

def overlay_cam_on_image(img: Image.Image, cam_map: np.ndarray) -> Image.Image:
    """
    Overlay the CAM heatmap on the original image.
    img: PIL Image (original image).
    cam_map: 2D numpy array (CAM heatmap) with values in [0,1].
    Returns: PIL Image with heatmap overlay.
    """
    # Resize heatmap to image size
    H, W = img.size[1], img.size[0]  # PIL size is (width, height)
    heatmap = (cm.jet(cam_map) * 255).astype(np.uint8)[:, :, :3]   # apply JET colormap
    heatmap = Image.fromarray(heatmap).resize((W, H), resample=Image.BILINEAR)
    heatmap = np.array(heatmap)
    # Overlay heatmap onto the original image
    img_np = np.array(img.convert('RGB'))
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)      # blend heatmap with image
    return Image.fromarray(overlay)

def compute_metrics(model, data_loader):
    """
    Compute classification metrics (accuracy) and optionally return CAM overlays for a few samples.
    """
    model.eval()
    correct, total = 0, 0
    cam_examples = []
    for batch_idx, (images, labels) in enumerate(data_loader):
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        # Store CAMs for first few batches
        if batch_idx < 1:  # e.g., take CAMs from the first batch as examples
            for i in range(min(5, labels.size(0))):
                cam_map = generate_cam(model, images[i], class_idx=int(preds[i]))
                # Assume images are normalized; convert back to PIL for visualization
                img_array = (images[i].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)  # rough un-normalization (if originally 0-1 scaled)
                img_pil = Image.fromarray(img_array)
                cam_overlay = overlay_cam_on_image(img_pil, cam_map)
                cam_examples.append((int(labels[i]), int(preds[i]), cam_overlay))
    accuracy = 100.0 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy, cam_examples
