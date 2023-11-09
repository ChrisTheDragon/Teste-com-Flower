import torch
from torch import nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import Redes_Neurais as M
import aux
from PIL import Image
import os
import seaborn as sn
import matplotlib.pyplot as plt