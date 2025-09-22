import argparse
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DualBranchSwinCNNClassifier

# If the original dual-Swin model is needed as teacher, ensure its class is accessible:
try:
    from model import DualBranchSwinTinyClassifier  # original full model architecture
except ImportError:
    DualBranchSwinTinyClassifier = None


def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------
    # Dataset & DataLoader setup
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # Resize images to 224x224
        transforms.ToTensor()
    ])

    # Point to your dataset folders
    train_dataset = datasets.ImageFolder(root="./data/train", transform=transform)
    val_dataset = datasets.ImageFolder(root="./data/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ----------------------------
    # Model setup
    # ----------------------------
    model = DualBranchSwinCNNClassifier(num_classes=args.num_classes, pretrained=True).to(device)

    # Teacher model (if distillation)
    teacher_model = None
    if args.teacher_path:
        if DualBranchSwinTinyClassifier is not None:
            teacher_model = DualBranchSwinTinyClassifier(num_classes=args.num_classes, pretrained=False).to(device)
        else:
            teacher_model = DualBranchSwinCNNClassifier(num_classes=args.num_classes, pretrained=False).to(device)
        checkpoint = torch.load(args.teacher_path, map_location=device)
        teacher_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                T = args.kd_temperature
                soft_loss = F.kl_div(
                    F.log_softmax(outputs / T, dim=1),
                    F.softmax(teacher_outputs / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)
                hard_loss = criterion(outputs, labels)
                loss = args.kd_alpha * soft_loss + (1 - args.kd_alpha) * hard_loss
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed.")

    # ----------------------------
    # Optional pruning
    # ----------------------------
    if args.prune_fraction > 0:
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=args.prune_fraction)
                prune.remove(module, 'weight')
        print(f"Pruned {args.prune_fraction*100:.0f}% of weights in Conv2d/Linear layers.")

    torch.save(model.state_dict(), args.output_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2, help='Number of target classes')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--teacher_path', type=str, default=None, help='Path to teacher model weights for distillation')
    parser.add_argument('--kd_alpha', type=float, default=0.5, help='Blend factor for distillation loss vs true loss')
    parser.add_argument('--kd_temperature', type=float, default=4.0, help='Temperature for softening logits in distillation')
    parser.add_argument('--prune_fraction', type=float, default=0.0, help='Fraction of weights to prune after training (0 to disable)')
    parser.add_argument('--output_model', type=str, default='student_model.pth', help='Path to save the trained model')
    args = parser.parse_args()
    train_model(args)
