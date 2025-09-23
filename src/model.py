import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from timm import create_model

class DualBranchSwinCNNClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # CNN (ResNet18)
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn.fc = nn.Identity()
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
        self.swin.head = nn.Identity()
        self.swin_pool = nn.AdaptiveAvgPool2d(1)

        # infer fused width and build head
        in_features = self._infer_fused_dim()
        self.fc = nn.Linear(in_features, num_classes)

    def _swin_feats(self, x):
        if hasattr(self.swin, "forward_features"):
            f = self.swin.forward_features(x)
        else:
            f = self.swin(x)
        if f.ndim == 4:     # [B,C,H,W]
            f = self.swin_pool(f).flatten(1)
        elif f.ndim == 3:   # [B,N,C]
            f = f.mean(dim=1)
        return f            # [B,C]

    @torch.no_grad()
    def _infer_fused_dim(self):
        self.eval()
        dummy = torch.zeros(1, 3, 224, 224)
        c = self.cnn(dummy)            # [1,C1]
        s = self._swin_feats(dummy)    # [1,C2]
        return c.shape[1] + s.shape[1]

    def forward(self, x):
        c = self.cnn(x)
        s = self._swin_feats(x)
        return self.fc(torch.cat([c, s], dim=1))

class DualBranchSwinTinyClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
        self.swin.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        with torch.no_grad():
            d = torch.zeros(1,3,224,224)
            f = self.swin.forward_features(d) if hasattr(self.swin,"forward_features") else self.swin(d)
            if f.ndim == 4: f = self.pool(f).flatten(1)
            elif f.ndim == 3: f = f.mean(dim=1)
        self.fc = nn.Linear(f.shape[1], num_classes)

    def forward(self, x):
        f = self.swin.forward_features(x) if hasattr(self.swin,"forward_features") else self.swin(x)
        if f.ndim == 4: f = self.pool(f).flatten(1)
        elif f.ndim == 3: f = f.mean(dim=1)
        return self.fc(f)
