import os
import json
import os.path as osp 
import cv2
import numpy as np

from helpers import *
from metrics import boundary_f1_score

input_dir = '/HDD/datasets/public/mini_cityscapes/gtFine/train/aachen'
img_file = osp.join(input_dir, 'aachen_000000_000019_gtFine_labelIds.png')
json_file = osp.join(input_dir, 'aachen_000000_000019_gtFine_polygons.json')
output_dir = '/HDD/research/boundary_aware/etc/noised_boundary'

classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole", 
    "traffic light", "traffic sign", "vegetation", "terrain", "sky", 
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

class2label = {_class: label for label, _class in enumerate(classes)}

if not osp.exists(output_dir):
    os.mkdir(output_dir)

with open(json_file, 'r') as jf:
    anns = json.load(jf)
    
image_height = anns['imgHeight']
image_width = anns['imgWidth']

mask = np.zeros((image_height, image_width), dtype=np.uint8)
noisy_mask = np.zeros((image_height, image_width), dtype=np.uint8)
for obj in anns['objects']:
    label = obj['label']
    if label not in class2label:
        continue
    polygon = obj['polygon']
    
    noisy_polygon = perturb_polygon(polygon, ratio=0.5, max_shift=5)    
    cv2.fillPoly(noisy_mask, [np.array(noisy_polygon, np.int32)], class2label[label]*255/len(class2label))
    cv2.fillPoly(mask, [np.array(polygon, np.int32)], class2label[label]*255/len(class2label))
    
cv2.imwrite(osp.join(output_dir, 'mask.png'), mask)
cv2.imwrite(osp.join(output_dir, 'noisy_mask.png'), noisy_mask)

f1_score = boundary_f1_score(mask, noisy_mask, dilation=1)
print(f1_score)
    
    



    

