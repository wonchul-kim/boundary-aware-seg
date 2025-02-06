import numpy as np
import cv2 

def boundary_f1_score(pred_mask, gt_mask, dilation=1):
    """ Compute boundary F1-score by comparing edge regions. """
    kernel = np.ones((dilation, dilation), np.uint8)
    
    pred_edges = cv2.Canny(pred_mask.astype(np.uint8) * 255, 100, 200)
    gt_edges = cv2.Canny(gt_mask.astype(np.uint8) * 255, 100, 200)
    
    tp = np.sum(np.logical_and(pred_edges, gt_edges))
    fp = np.sum(np.logical_and(pred_edges, np.logical_not(gt_edges)))
    fn = np.sum(np.logical_and(np.logical_not(pred_edges), gt_edges))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return f1
