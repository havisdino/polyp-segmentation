import torch
from torchvision import io, transforms
import matplotlib.pyplot as plt
import numpy as np


def decode_and_normalize_image(raw_byte):
    raw_byte = torch.ByteTensor(list(raw_byte))
    img = io.decode_image(raw_byte).unsqueeze(0)[:, :3, ...]
    return img / 255.0


def merge(image, mask):
    image = image.squeeze().permute(1, 2, 0).numpy()
    prob_mask = mask.squeeze(0).numpy()
    bin_mask = np.round(prob_mask)
    heatmap = plt.cm.jet(prob_mask)[..., :3]
    segmentation = (image + heatmap) * 0.5
    return image, segmentation, prob_mask, bin_mask
    
    