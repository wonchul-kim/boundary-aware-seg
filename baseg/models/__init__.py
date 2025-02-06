import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm  # For DINOv2 model
import numpy as np
import cv2
from torchvision.models.segmentation import deeplabv3_resnet50  # Can replace with SAM2

# Load DINOv2 model (pretrained backbone)
class FeatureExtractor(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224.dino"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)  # DINOv2 model
    def forward(self, x):
        return self.model(x)  # Extract features



class BoundaryRefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))  # Output refined boundary


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.low_res_branch = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.high_res_branch = nn.Conv2d(in_channels, 128, kernel_size=1)

    def forward(self, deep_features, early_features):
        low_res_feat = self.low_res_branch(deep_features)
        high_res_feat = self.high_res_branch(early_features)
        return torch.cat([low_res_feat, high_res_feat], dim=1)  # Fused representation


class BoundaryDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class BoundaryAwareSegmentation(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.feature_extractor = backbone  # Use DINOv2/SAM2
        self.seg_head = deeplabv3_resnet50(pretrained=True)  # DeepLabV3+ for segmentation
        self.boundary_refine = BoundaryRefinementModule()
        self.boundary_discriminator = BoundaryDiscriminator()

    def forward(self, x):
        features = self.feature_extractor(x)  # Extract features
        segmentation_mask = self.seg_head(x)['out']  # Predict segmentation
        refined_boundary = self.boundary_refine(segmentation_mask)  # Refine boundaries
        
        return segmentation_mask, refined_boundary


if __name__ == '__main__':
    model = BoundaryAwareSegmentation()