#we will be writing all the utility functions for Plung and play iteartive optimization

import importlib
#import parameters from file nn_denoiser_parameter.py
import denoisers.nn_denoiser_parameter
importlib.reload(denoisers.nn_denoiser_parameter)
from denoisers.nn_denoiser_parameter import *
#patch aggregator
import denoisers.patch_aggregator
importlib.reload(denoisers.patch_aggregator)
from denoisers.patch_aggregator import *

#import utils
import scripts_orthogonal_training.utils
importlib.reload(scripts_orthogonal_training.utils)
importlib.reload(scripts_orthogonal_training)
from scripts_orthogonal_training.utils import *

#import image quality metrics
import scripts_orthogonal_training.image_quality_metrics
importlib.reload(scripts_orthogonal_training.image_quality_metrics)
from scripts_orthogonal_training.image_quality_metrics import *
#import dataset functions
import scripts_orthogonal_training.training_dataset
importlib.reload(scripts_orthogonal_training.training_dataset)
from scripts_orthogonal_training.training_dataset import *
#import models
import scripts_orthogonal_training.models
importlib.reload(scripts_orthogonal_training.models)
#import all from models
from scripts_orthogonal_training.models import *
from scripts_orthogonal_training.models import BaselineModelTanh, BaselineModelTanhTanh
#imports 
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




#get the transform
transform = get_transform(num_channels=num_channels, mean=mean, std=std)
#we will use the BaselineModelTanh class
baseline_model = BaselineModelTanh(num_layers=NUM_LAYERS)
baseline_model = to_double(baseline_model, data_type)
#call model_to_device function
# model_to_device(baseline_model, device)


#we write a function to create a patch denoiser, i.e. load the preatrineed model into the created model
baseline_model = BaselineModelTanh(num_layers=NUM_LAYERS)

def load_model(device, net = baseline_model, pre_trained_model_path = model_path):
    model_to_device(net, device)
    #make double datatype
    net = to_double(net, data_type=data_type)
    
    net.load_state_dict(torch.load(pre_trained_model_path))
    net.to(device)
    #model loaded --------------------------------
    #set modle to eval mode
    net.eval()
    #return the model
    return net


#we write a patch denoier function using the loaded model
#input: patch (numpy ndarray)
#output: denoised patch
#processing: we will first convert the numpy ndarray to pil patch to tensor using the transform, then we will add a batch dimension
#then we will convert the tensor to double datatype and send it to device
#then we will pass the tensor to the model and get the output
#then we will convert the output to numpy and remove the batch dimension

def nn_patch_denoiser(patch, net, device):
    #convert patch from numpy to PIL, if its numpy
    #else, if its PIL or torch tensor, do nothing
    # if isinstance(patch, np.ndarray):
    #     patch = Image.fromarray(patch)
    # #convert patch from PIL to tensor using the transform
    # patch = transform(patch)

    #if not a tensor, convert to tensor, 
    #if numpy, convert then to tensor
    if not isinstance(patch, torch.Tensor):
        #convert to tensor
        patch = torch.from_numpy(patch)
    #add batch dimension
    patch = patch.unsqueeze(0)
    #convert to data type data_type this torch tensor
    patch = patch.to(data_type)
    #send to device
    patch = patch.to(device)

    ##debug----------------
    #print the min and max of the patch
    # print('max of noisy patch: ', torch.max(patch))
    # print('min of noisy patch: ', torch.min(patch))

    ##debug----------------
    #pass the patch to the model
    with torch.no_grad():
        output = net(patch)
    #convert output to numpy
    output = output.cpu().numpy()
    #remove batch dimension
    output = output[0]

    #debug----------------
    #print the min and max of the output
    # print('max of output: ', np.max(output))
    # print('min of output: ', np.min(output))
    #debug----------------


    #return the output
    return output


#we will here denoise the barbara image using the aggregator , and our patch denoiser would be the nn_denoiser

def nn_denoiser(noisy_image , device, baseline_model=baseline_model, model_path =model_path, current_range= (0,255),  \
                transform = transform , patch_size = patch_size, s = 1 ):
    #call load model function
    model = load_model(device = device, net = baseline_model, pre_trained_model_path = model_path)
    
    #noisy_image should numpy array, else convert it to numpy array
    if not isinstance(noisy_image, np.ndarray):
        noisy_image = np.array(noisy_image)
    #if there is no value > 1, then print a warning that 'scale might not be 0-255'
    if np.max(noisy_image) <= 1:
        print('Warning: scale might not be 0-255')

    #get the min and max current range
    current_range_min = current_range[0]
    current_range_max = current_range[1]
    #get the min and max of the noisy image
    noisy_image_min = np.min(noisy_image)
    #if the min of teh noisy image is less than the current range min, then update the noisy image min
    if noisy_image_min < current_range_min:
        noisy_image_min = current_range_min
    noisy_image_max = np.max(noisy_image)
    #if the max of teh noisy image is greater than the current range max, then update the noisy image max
    if noisy_image_max > current_range_max:
        noisy_image_max = current_range_max
    #now we will convert it to PIL image
    noisy_image = Image.fromarray(noisy_image)

    #we will get the transform function and the transformed image
    #now we will convert it to tensor
    noisy_image = transform(noisy_image)
    #remove the batch dimension, so that we have only 2 dimension image
    noisy_image = noisy_image.squeeze(0)
    #convert to numpy array
    noisy_image = np.array(noisy_image)

    #we will construct the denoiser function kwargs
    denoiser_function_kwargs = {'net': model, 'device': device}
    #call the aggregator function
    denoised_image = patch_aggregator(image = noisy_image, denoiser_function= nn_patch_denoiser, \
                                        denoiser_function_kwargs = denoiser_function_kwargs, \
                                        patch_size = patch_size, s = s)
    #now we will convert it to numpy array
    denoised_image = np.array(denoised_image)
    #get the min and max of the denoised image
    denoised_image_min = np.min(denoised_image)
    denoised_image_max = np.max(denoised_image)
    #we will resale_range the denoised image using the function rescale_range based on 
    # the min and max of the noisy image and the min and max of the denoised image
    denoised_image = rescale_range(denoised_image, (denoised_image_min, denoised_image_max) ,\
                                    (noisy_image_min, noisy_image_max))

   
    return denoised_image


