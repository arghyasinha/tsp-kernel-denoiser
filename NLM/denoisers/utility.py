#we will be writing all the utility functions for Plung and play iteartive optimization

import importlib
#import utils
import scripts_orthogonal_training.utils
importlib.reload(scripts_orthogonal_training.utils)
importlib.reload(scripts_orthogonal_training)
from scripts_orthogonal_training.utils import *

#import image quality metrics
import scripts_orthogonal_training.image_quality_metrics
importlib.reload(scripts_orthogonal_training.image_quality_metrics)
from scripts_orthogonal_training.image_quality_metrics import *

#imports 
#import denoise_nl_means from skimage
from skimage.restoration import denoise_nl_means, estimate_sigma
#BM3D
import bm3d
##### libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import skimage
from skimage import io
from skimage import color
from skimage import img_as_float
import skimage.measure as measure
from skimage import metrics
from skimage import img_as_ubyte
import os
from torch.autograd import Variable
import time
import argparse
import random

import matplotlib.pyplot as plt
from torchsummary import summary
#imports for dataset and dataloader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
import cv2
#math
import math
#import copy
import copy


#seed the random number generator
seed =42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#add one directory up to the path
import sys
sys.path.append('../')

#set the print options
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

