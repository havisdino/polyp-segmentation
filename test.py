import torch
from torchvision import io, transforms
import matplotlib.pyplot as plt
from model import *
from torchview import draw_graph


net = UNet()
state_dict = torch.load('trained_model/unet.pt', 'cpu')
net.load_state_dict(state_dict)
net.eval()

x = torch.randn(1, 3, 256, 256)
