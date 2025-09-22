import torch
import torch.nn as nn
from torchvision import models
import timm

class DualBranchSwinCNNClassifier(nn.Module):
    """
    Dual-branch classifier with one Swin Transformer (global features)
    and one CNN (ResNet18) branch (local features). Outputs class logits.
    """
    def __init__(self, num_classes, pretrained=True):
        super(DualBranchSwinCNNClassifier, self).__init__()
        # Swin Transformer Tiny branch (pretrained on ImageNet)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', 
                                      pretrained=pretrained, num_classes=0)
        # CNN branch: ResNet18 (pretrained on ImageNet)
        self.cnn = models.resnet18(pretrained=pretrained)
        # Remove ResNet18's final FC layer to use as feature extractor
        self.cnn.fc = nn.Identity()
        # Feature dimensions from each branch
        swin_feat_dim = getattr(self.swin, 'num_features', 768)  # Swin-Tiny output dim (~768)
        cnn_feat_dim = 512  # ResNet18 output dim after global pooling
        # Classification head on concatenated features
        self.fc = nn.Linear(swin_feat_dim + cnn_feat_dim, num_classes)

    def forward(self, x):
        # **Swin Transformer branch** – get feature map and global pool
        swin_featmap = self.swin.forward_features(x)       # shape [B, H, W, C] for Swin
        if swin_featmap.dim() == 4 and swin_featmap.shape[1] != getattr(self.swin, 'num_features', swin_featmap.shape[-1]):
            # If output is [B, H, W, C], permute to [B, C, H, W]
            swin_featmap = swin_featmap.permute(0, 3, 1, 2)
        swin_vec = swin_featmap.mean(dim=(2, 3))           # global average pool -> [B, C]

        # **CNN branch (ResNet18)** – forward to get final conv feature map
        x_cnn = self.cnn.conv1(x)
        x_cnn = self.cnn.bn1(x_cnn)
        x_cnn = self.cnn.relu(x_cnn)
        x_cnn = self.cnn.maxpool(x_cnn)
        x_cnn = self.cnn.layer1(x_cnn)
        x_cnn = self.cnn.layer2(x_cnn)
        x_cnn = self.cnn.layer3(x_cnn)
        x_cnn = self.cnn.layer4(x_cnn)                    # final conv feature map [B, 512, H_c, W_c]
        cnn_vec = x_cnn.mean(dim=(2, 3))                  # global average pool -> [B, 512]

        # **Feature fusion and classification**
        combined_feat = torch.cat([swin_vec, cnn_vec], dim=1)
        logits = self.fc(combined_feat)
        return logits

    # (Optional) A helper method could be added to retrieve feature maps for CAM if needed.
