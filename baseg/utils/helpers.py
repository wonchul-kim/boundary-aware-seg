import random
import numpy as np

def perturb_polygon(polygon, ratio=1, max_shift=3):
    """
    Add small random offsets to polygon vertices.
    
    Args:
        polygon: List of (x, y) coordinates.
        max_shift: Maximum pixel shift.
    
    Returns:
        Noisy polygon with perturbed boundaries.
    """
    noisy_polygon = []
    for x, y in polygon:
        if random.random() <= ratio:
            x = x + random.randint(-max_shift, max_shift)
            y = y + random.randint(-max_shift, max_shift)

        noisy_polygon.append((x, y))
    
    return noisy_polygon


def get_adaptive_max_shift(epoch, total_epochs, max_shift=3):
    """
    Reduce noise over time
    
    - Early training → More noise (forces boundary generalization).
    - Late training → Less noise (refines final segmentation accuracy).
    """
    
    return max(1, int(max_shift * (1 - epoch / total_epochs)))

 