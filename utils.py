import torch
from torchvision import io, transforms


def decode_and_normalize_image(raw_byte):
    raw_byte = torch.ByteTensor(list(raw_byte))
    img = io.decode_image(raw_byte)
    return img[..., :3] / 255.0