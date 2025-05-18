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


########## BM3D
#we will build a wrapper fi=unction for BM3D denoiser
def BM3D(noisy_image, current_range, sigma_psd, mode='slow'):
    target_range = (0, 1)
    #we will convert the sigma_psd to the range 0 to 1
    #currently sigma_psd is in the range current_range
    #we want appropriate conversion of sigma_sd to the range 0 to 1
    #print new line
    # print()
    #print sigma_psd
    # print('sigma_psd: ', sigma_psd)
    # divide sigma_psd by the range of the current range and multiply by the range of the target range
    sigma_psd = (sigma_psd / (current_range[1] - current_range[0])) * (target_range[1] - target_range[0])
    #print sigma_psd
    # print('sigma_psd: ', sigma_psd)
    #we will first have to convert the image to teh range 0 to 1, where BM3D expects the image to be
    #we will use the function rescale_range
    #the function rescale_range takes in the image, the current range of the image, and the desired range
    #the current range is the parameter current_range, target range is (0,1) as defined
    noisy_image = rescale_range(noisy_image, current_range, target_range)
    #now based on fast or slow , we will use the stage_arg
    if mode == 'fast':
        stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING
    else:
        stage_arg = bm3d.BM3DStages.ALL_STAGES
    #call
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd = sigma_psd, stage_arg = stage_arg)
    #now we will convert the denoised image back to the original range
    denoised_image = rescale_range(denoised_image, target_range, current_range)
    #return the denoised image
    return denoised_image
