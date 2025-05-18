import importlib

#import parameters from file nn_denoiser_parameter.py
import denoisers.nn_denoiser_parameter
importlib.reload(denoisers.nn_denoiser_parameter)
from denoisers.nn_denoiser_parameter import *


#import config
#--------------------------------------
import config.config_template
importlib.reload(config.config_template)
importlib.reload(config)
from config.config_template import *
########################################
########################################

#import models
import scripts_orthogonal_training.models
importlib.reload(scripts_orthogonal_training.models)
#import all from models
from scripts_orthogonal_training.models import *
from scripts_orthogonal_training.models import BaselineModelTanh, BaselineModelTanhTanh
#import dataset functions
import scripts_orthogonal_training.training_dataset
importlib.reload(scripts_orthogonal_training.training_dataset)
from scripts_orthogonal_training.training_dataset import *
#import image quality metrics
import scripts_orthogonal_training.image_quality_metrics
importlib.reload(scripts_orthogonal_training.image_quality_metrics)
from scripts_orthogonal_training.image_quality_metrics import *
#import parameters
import parameters.parameters
importlib.reload(parameters.parameters)
from parameters.parameters import *
#import utils
import scripts_orthogonal_training.utils
importlib.reload(scripts_orthogonal_training.utils)
importlib.reload(scripts_orthogonal_training)
from scripts_orthogonal_training.utils import *
#import orthogonalization 
import scripts_orthogonal_training.orthogonalization
importlib.reload(scripts_orthogonal_training)
importlib.reload(scripts_orthogonal_training.orthogonalization)
from scripts_orthogonal_training.orthogonalization import *
#import analysis
import scripts_orthogonal_training.analysis
importlib.reload(scripts_orthogonal_training)
importlib.reload(scripts_orthogonal_training.analysis)
from scripts_orthogonal_training.analysis import *
# -------------------------------------------------------------------------------------------------

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
#opencv
import cv2

#import garbage collector
import gc
#math
import math
#import copy
import copy
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
   

#seed the random number generator
seed =42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
#add one directory up to the path
import sys
sys.path.append('../')
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=10_000)
torch.set_printoptions(precision=10)
np.set_printoptions(precision=10)

pad_mode_multiple = 'reflect'
pad_mode_border = 'wrap'

#we will create a padding function that will pad zeroes to the image at the end of rows and columns
#we will make the image a multiple of patch_size
#thus we will add zeroes to the image at the end of rows and columns

def pad_image_multiple(image, patch_size=patch_size, pad_mode=pad_mode_multiple):
    #get the shape of the image
    image_shape = image.shape
    #get the number of rows and columns
    num_rows = image_shape[0]
    num_cols = image_shape[1]
    #get the number of rows and columns to be padded
    #if the number of rows and columns are already a multiple of patch_size, then we will not pad
    #else we will pad
    if num_rows % patch_size == 0:
        num_rows_to_pad = 0
    else:
        num_rows_to_pad = patch_size - (num_rows % patch_size)
    if num_cols % patch_size == 0:
        num_cols_to_pad = 0
    else:
        num_cols_to_pad = patch_size - (num_cols % patch_size)
    # test_pad_image()nt('num_cols%patch_size: ', num_cols%patch_size)
    #debug----------------------------------------------------------------------------
    #pad the image
    padded_image = np.pad(image, ((0, num_rows_to_pad), (0, num_cols_to_pad)), pad_mode)
    return padded_image
#we will write a function that will pad  one row on top and one row at the bottom of the image
#one column size of patch size on the left and one column size of patch size on the right
#thus we will reate a border of patch size on all sixdes of th image

def pad_border_patch_size(image, patch_size=patch_size, pad_mode=pad_mode_border):
    #get the shape of the image
    image_shape = image.shape
    #get the number of rows and columns
    num_rows = image_shape[0]
    num_cols = image_shape[1]
    #pad the image
    padded_image = np.pad(image, ((patch_size, patch_size), (patch_size, patch_size)), pad_mode)
    return padded_image
#write a function ot return an image with border padded and pad_image to be multiple of patch_size

def pad_image(image, patch_size=patch_size, pad_mode_multiple=pad_mode_multiple, pad_mode_border=pad_mode_border):
    #pad the image to be a multiple of patch_size
    padded_image = pad_image_multiple(image, patch_size=patch_size, pad_mode=pad_mode_multiple)
    #pad the image to have a border of patch_size
    padded_image = pad_border_patch_size(padded_image, patch_size=patch_size, pad_mode=pad_mode_border)
    return padded_image

#### remove padding
#we will write a function that will remove border padding from the image, meaning
#we will remove the border of patch_size from all sides of the image
#it will remove rows and columns from the top, bottom, left and right of the image
#of pixel size patch_size

def remove_border_padding(image, patch_size=patch_size):
    #get the shape of the image
    image_shape = image.shape
    #get the number of rows and columns
    num_rows = image_shape[0]
    num_cols = image_shape[1]
    #remove the border padding
    image = image[patch_size:num_rows - patch_size, patch_size:num_cols - patch_size]
    return image
#we will remove the padding from the image to make it a multiple of patch_size
#this padding is added to the image to make it a multiple of patch_size
#the padding is added at the end of the rows and columns of the image
#so we just have to extract the image from the top left corner of the padded image

#input: image, image_shape
#output: image

def remove_padding_multiple(image, image_shape):
    #get the number of rows and columns
    num_rows = image_shape[0]
    num_cols = image_shape[1]
    #remove the padding
    image = image[0:num_rows, 0:num_cols]
    return image
#we write a single function that will remove padding from the image
#1. first it will remove the border padding
#2. then it will remove the padding to make the image a multiple of patch_size

def remove_padding(image, image_shape, patch_size=patch_size):
    #remove the border padding
    image = remove_border_padding(image, patch_size=patch_size)
    #remove the padding to make the image a multiple of patch_size
    image = remove_padding_multiple(image, image_shape)
    return image

#we will create a function that will take a noisy image and denoise it using the patch denoiser
#we will move with a stride: s

def patch_aggregator(image, denoiser_function, denoiser_function_kwargs = {}, patch_size= patch_size, pad_mode_multiple = \
                     pad_mode_multiple, pad_mode_border = pad_mode_border, s=1):
    #we will get the shape of the image
    image_shape = image.shape
    #get the number of rows and columns
    num_rows = image_shape[0]
    num_cols = image_shape[1]

    ## debug----------------------------------------------------------------------------------------
    # print('image_shape: ', image_shape)
    # print('num_rows: ', num_rows)
    # print('num_cols: ', num_cols)
    # #print image
    # print('image: ', image)
    # print('-' * 50)

    ## debug----------------------------------------------------------------------------------------

    #pad the image
    padded_image = pad_image(image, patch_size=patch_size, pad_mode_multiple=pad_mode_multiple, \
                                pad_mode_border=pad_mode_border)


    #get shape of padded image
    padded_image_shape = padded_image.shape
    #get the number of rows and columns
    padded_num_rows = padded_image_shape[0]
    padded_num_cols = padded_image_shape[1]

    #assert that the padded image is a multiple of patch_size
    assert padded_num_rows % patch_size == 0, 'padded image rows is not a multiple of patch_size'
    assert padded_num_cols % patch_size == 0, 'padded image cols is not a multiple of patch_size'

    #debug----------------------------------------------------------------------------------------
    # #print padded image shape
    # print('padded_image_shape: ', padded_image_shape)
    # print('padded_num_rows: ', padded_num_rows)
    # print('padded_num_cols: ', padded_num_cols)

    # #print padded image
    # print('padded_image: ', padded_image)
    # print('-' * 50)
    #debug----------------------------------------------------------------------------------------

    #debug----------------------------------------------------------------------------------------
    #we remove the padding from the image
    # unpadded_image = remove_padding(padded_image, image_shape, patch_size=patch_size)
    # #print unpadded image
    # print('unpadded_image: ', unpadded_image)
    # print('-' * 50)
    
    #debug----------------------------------------------------------------------------------------
    
    #if patch size =k, patch height = k and patch width = k
    #then patch size = k*k
    #thus number of overlapping patches = (k/s)*(k/s)
    #so one pixel will be part of (k/s)*(k/s) patches
    #we will then average the pixel values of all the patches that it is part of
    #so we will make (k/s)*(k/s) channels as one for each stride in a patch
    #later we will sum the channels and divide by (k/s)*(k/s) to get the average

    #we will call patch_denioiser on each patch and then store the returned denoised patch to the appropriate\
    #location in the channel it is part of

    #store k/s into variable overlaps
    #assert that patch_size is a multiple of s
    assert patch_size % s == 0, 'patch_size is not a multiple of s'
    overlaps = int(patch_size/s)
    #create an output image of shape(padde_num_rows, padded_num_cols, overlaps*overlaps)
    #fill it with zeroes
    temp_image = np.zeros((padded_num_rows, padded_num_cols, overlaps*overlaps))
    #we will focus on strides in first patch, as that sqaure will have k/s*k/s strides
    #and we will move the patch by s pixels in each stride in horizontal and vertical direction
    #
    #we will start the stride firstly by (0,0) pixel and read the patch of size patch_sizexpatch_size
    #and then we will move patch_size pixels in both horizontal and vertical direction
    #and fill the 0th channel of the output image with the denoised patch

    #then we will select the pixel(0, s) and read the patch of size patch_sizexpatch_size
    #and then we will move patch_size pixels in both horizontal and vertical direction
    #and fill the 1st channel of the output image with the denoised patch
    #and so on
    #so we have (k/s)*(k/s) starting pixels and we will read the patch of size patch_sizexpatch_size
    #and then we will move patch_size pixels in both horizontal and vertical direction, i.e 
    #non-overlapping patches for each channel
    #and populate the channel with the denoised patch

    #we will stop on second last column as in last column we can not take stride of any size
    #as we will move out of the image
    #similarly in the last row we will stop on second last row as we can not take stride of any size
    #as we will move out of the image

    #get the number of columns and rows in the padded image
    #assert that the padded image is a multiple of patch_size
    assert padded_num_rows % patch_size == 0, 'padded_num_rows is not a multiple of patch_size'
    num_rows_non_overlap = int(padded_num_rows/patch_size)
    assert padded_num_cols % patch_size == 0, 'padded_num_cols is not a multiple of patch_size'
    num_cols_non_overlap = int(padded_num_cols/patch_size)

    #we will loop through the starting pixels in first patch
    for s_r in range(overlaps):
        for s_c in range(overlaps):
            #we will move non-overlapping patches of size patch_sizexpatch_size
            #from this starting pixel and save in channel s_r*overlaps + s_c

            #we will loop through the non-overlapping patches
            #except the last patch in each row and column

            #iterate through the rows: 0 to num_rows_non_overlap - 1
            for r in range(num_rows_non_overlap - 1):
                for c in range(num_cols_non_overlap - 1):
                    #get the starting pixel
                    start_r = r * patch_size + s_r * s
                    start_c = c * patch_size + s_c * s
                    #get the patch
                    patch = padded_image[start_r:start_r + patch_size, start_c:start_c + patch_size]
                    #denoise the patch
                    denoised_patch = denoiser_function(patch, **denoiser_function_kwargs)
                    #save the denoised patch in the appropriate channel
                    temp_image[start_r:start_r + patch_size, start_c:start_c + patch_size, s_r*overlaps + s_c]\
                                  = denoised_patch
        ## debug----------------------------------------------------------------------------------------
        #             break
        #         break
        #     break
        # break
        ## debug----------------------------------------------------------------------------------------


    #we will now get an output image of shape (padded_num_rows, padded_num_cols) by summing the channels
    #and dividing by overlaps*overlaps
    output_image = np.sum(temp_image, axis=2)/(overlaps*overlaps)

    #delete the temp_image
    del temp_image, overlaps, num_rows_non_overlap, num_cols_non_overlap, padded_num_cols, padded_num_rows

    #debug----------------------------------------------------------------------------------------
    # #print output image shape
    # print('output_image_shape: ', output_image.shape)
    # #print output image max and min
    # print('output_image_max: ', np.max(output_image))
    # print('output_image_min: ', np.min(output_image))
    # #print output image
    # print('output_image: ', output_image)
    #debug----------------------------------------------------------------------------------------
    
    #we will call remove_padding function to remove the padding
    output_image = remove_padding(output_image, image_shape, patch_size)
    #debug----------------------------------------------------------------------------------------
    # #print the variable overlaps
    # print('overlaps: ', overlaps)
    # #print the variable image_shape
    # print('image_shape: ', image_shape)
    # #print output image shape
    # print('output_image_shape: ', output_image.shape)
    # #print output image max and min
    # print('output_image_max: ', np.max(output_image))
    # print('output_image_min: ', np.min(output_image))
    # #print output image
    # print('output_image: ', output_image)
    # print('-' * 50)
    #debug----------------------------------------------------------------------------------------
    #return the output image
    return output_image
            

