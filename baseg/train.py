import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
import numpy as np
import cv2
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
])

train_dataset = Cityscapes(root='/HDD/datasets/public/cityscapes', split='train', mode='fine', target_type='semantic', transform=transform, target_transform=transform)
val_dataset = Cityscapes(root='/HDD/datasets/public/cityscapes', split='val', mode='fine', target_type='semantic', transform=transform, target_transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_small_patch16_224.dino'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
    
    def forward(self, x):
        return self.model(x)

feature_extractor = FeatureExtractor().cuda()

def perturb_boundary(gt_mask, max_shift=3):
    kernel = np.ones((3, 3), np.uint8)
    if np.random.rand() > 0.5:
        noisy_gt_mask = cv2.dilate(gt_mask.cpu().detach().numpy().astype(np.uint8), kernel, iterations=np.random.randint(1, max_shift+1))
    else:
        noisy_gt_mask = cv2.erode(gt_mask.cpu().detach().numpy().astype(np.uint8), kernel, iterations=np.random.randint(1, max_shift+1))
    return torch.tensor(noisy_gt_mask, dtype=torch.float32)

model = BoundaryAwareSegmentation(feature_extractor).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for images, gt_masks in tqdm(train_loader):
        images, gt_masks = images.cuda(), gt_masks.cuda()
        noisy_gt_masks = torch.stack([perturb_boundary(mask) for mask in gt_masks])

        optimizer.zero_grad()
        pred_masks, refined_boundary = model(images)
        loss = loss_fn(pred_masks, noisy_gt_masks.long())
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')