## imports
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import os
import cv2
import scipy
from scipy import ndimage
from scipy import signal
from scipy import misc
from scipy import linalg

    
#we will use GPU for matrix operations
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F

#enable cudnn
cudnn.benchmark = True
# #cuda cache clear
torch.cuda.empty_cache()

#we will use tensorboard to visualize the results
from torchsummary import summary
#import tenserboard
from torch.utils.tensorboard import SummaryWriter

#device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#imports
#import utils.py from parent directory
import sys
sys.path.append('../')

import importlib
#import utils
import utils
importlib.reload(utils)
from utils import *


#import necessary libraries
import numpy as np

import matplotlib.pyplot as plt
#import ticker
import matplotlib.ticker as ticker
#import mathplotlib
import matplotlib
import matplotlib.pyplot
#create writer for tensorboard
# writer = SummaryWriter(f'runs/' + experiment_name)

#get one directory to this notebook
import os
import sys
 

#seed all the random, python, numpy
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


#lets define hat

def compute_hat(s):
    #iterate dimension of s
    result = 1
    for i in range(len(s)):
        result = result * max(0, 1 - abs(s[i]))
    return result

def DSG_NLM_MATRIX( reference_image ,search_window = 7, patch_size=35, h=10):
#get the parameters of the NLM algorithm from the function set_parameters if they are not provided
#call the function set_parameters with sigma

    
    if len(reference_image.shape) == 3:
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    #debug----------------
    #print search window size, patch size, h
    # print("search window size is ", search_window)
    # print("patch size is ", patch_size)
    # print("h is ", h)
    #---------------------debug
    #get the dimensions of the noisy image 
    noisy_height, noisy_width = reference_image.shape
    #get the dimensions of the reference image
    ref_height, ref_width = reference_image.shape

    #initialize weight matrix, its size is the same as the noisy image
    #the outputweight matrix is of the shape ( (noisy_height x noisy_width), (noisy_height x noisy_width) )
    K = np.zeros((noisy_height * noisy_width, noisy_height * noisy_width))
    #we will build K as a corresponding kernal matrix when image is flattened
    #thus, the dimension of K is (noisy_height x noisy_width, noisy_height x noisy_width)\
    #each row will contain weights to be multiplied to the corresponding pixel in the noisy image (flattened)
    #

    # #debug-------------dest_x-i, dest_y-j
    # #print shape of K
    # print("shape of K is ", K.shape)


    # #debug-------------
    #set the size of the search window and the patch size threshold to the minimum of the 
    # height and width of the image
    if search_window > min(noisy_height, noisy_width):
        search_window = min(noisy_height, noisy_width)
        #print this warning
        print("Warning: Search window size is larger than the image size. Setting search window size to the minimum of the image size")
    if patch_size > min(ref_height, ref_width):
        patch_size = min(ref_height, ref_width)
        #print this warning
        print("Warning: Patch size is larger than the image size. Setting patch size to the minimum of the image size")
    #the search window size should be odd, if it is even, then we will subtract 1 from it, and print a warning
    if search_window % 2 == 0:
        search_window -= 1
        print("Warning: Search window size is even. Setting search window size to ", search_window)
    #the patch size should be odd, if it is even, then we will subtract 1 from it, and print a warning
    if patch_size % 2 == 0:
        patch_size -= 1
        print("Warning: Patch size is even. Setting patch size to ", patch_size)
    # #if patch size is greater than search window size, then we will set the patch size to the search window size
    # #and print a warning
    # if patch_size > search_window:
    #     patch_size = search_window
    #     print("Warning: Patch size is greater than search window size. Setting patch size to ", patch_size)
    
    #NOW we have odd, search_window and patch_size, and
    #search_window <= patch_size <= min(min(noisy_height, noisy_width), min(ref_height, ref_width))

    #we will write the code to build matrix for non-local means , but we have 2 images
    #the noisy_image is the one, which we will use to read the pixel values
    #the reference_image is the one, which we will use to compute weights

    #thus in the formula of NLM  denoised image at pixel (i,j) is given by
    # u(i,j) = sum( y \in window of size window_size around (i,j) ) w(i,j,x,y) * v(y) / 
    # sum( y \in window of size window_size around (i,j) ) w(i,j,x,y)

    #thus the weights are given by the reference image, and the noisy image is used to compute the denoised image
    #pad the noisy and the guide image with symmetric padding
    #noisy image is padded with symmetric padding of size (search_window-1)/2 in both directions


    # padded_noisy_image = np.pad(noisy_image, ((search_window-1)//2, (search_window-1)//2), 'symmetric').copy()
    #reference image is padded with symmetric padding of size (patch_size-1)/2 in both directions
    # padded_ref_image = np.pad(reference_image, ((patch_size-1)//2, (patch_size-1)//2), 'symmetric').copy()


    #now we will iterate over the noisy image, and compute the weights for each pixel
    #the 
    for i in range(noisy_height):
        for j in range(noisy_width):
            # we get the pixel location of current (i,j) pixel in the flattened image
            #thus tinstead of (i,j) we will use (i*noisy_width + j) to get the pixel location in the flattened image
            source_location = i*noisy_width + j
            #get the patch at the current pixel , but we will read this in the padded reference image
            #so for pixel location (i,j) in the noisy image, the corresponding patch center in teh 
            #padded reference image is (i+(patch_size-1)/2, j+(patch_size-1)/2)
            #and we will read the patch from the padded reference image
            # patch_current = padded_ref_image[\
            #     i+(patch_size-1)//2-(patch_size-1)//2:i+(patch_size-1)//2+(patch_size-1)//2+1,\
            #           j+(patch_size-1)//2-(patch_size-1)//2:j+(patch_size-1)//2+(patch_size-1)//2+1]
            #we will use the function : ndimage.interpolation.shift to get  a patch
            #from the reference image, centered at (i,j) in the noisy image of size patch_size x patch_size
            patch_current = ndimage.interpolation.shift(reference_image, (-(i-((patch_size-1)//2)), -(j-((patch_size-1)//2))),\
                                                         order=0, mode='reflect')[0:patch_size, 0:patch_size]
            
            #now we will extract patches from the search window around the current pixel
            #the search window is centered at the current pixel, and the size of the search window is search_window
            #loop over the pixels in the search window
            #we will use the noisy image only , and will loop over a window (search_window x search_window)
            #centered at the current pixel in the noisy image
            #so -(search_window-1)/2 to +(search_window-1)/2 in both directions
            #whenever we encounter a negative index, of the destination pixel, we will bring it as absolute value
            # to effectively get the symmetric padding
            # loop (-search_window//2, search_window//2+1) because we want to include the center pixel as well
            
            for dest_x in range( ( max(0, i-(search_window-1)//2) ), ( min(noisy_height, i+(search_window-1)//2+1) ) ):
                for dest_y in range( ( max(0, j-(search_window-1)//2) ), ( min(noisy_width, j+(search_window-1)//2+1) ) ):
                    #get the location of the destination pixel in the noisy image
            # for dest_x in range( ( max(0, i-(search_window)) ), ( min(noisy_height, i+(search_window)+1) ) ):
            #     for dest_y in range( ( max(0, j-(search_window)) ), ( min(noisy_width, j+(search_window)+1) ) ):
            #         #get the location of the destination pixel in the noisy image
                    target_location = dest_x*noisy_width + dest_y
            ######## FOR WEIGHTS OUTSIDE BOUNDARIES ########
            # for dest_distance_x in range(-(search_window-1)//2, (search_window-1)//2+1):
            #     for dest_distance_y in range(-(search_window-1)//2, (search_window-1)//2+1):
            #         #get the location of the destination pixel in the noisy image
            #         dest_x = i + dest_distance_x
            #         dest_y = j + dest_distance_y
            #         #if the destination pixel is outside the image, then we will bring it inside the image
            #         #by taking the absolute value
            #         if dest_x < 0:
            #             dest_x = abs(dest_x)
            #         if dest_y < 0:
            #             dest_y = abs(dest_y)
            #         if dest_x >= noisy_height:
            #             #if the destination pixel is outside the image, then we will bring it inside the image
            #             #by subtracting the distance it is outside the image from the image size
            #             distance_x_outside = dest_x - (noisy_height - 1)
            #             dest_x = noisy_height - distance_x_outside - 1
            #         if dest_y >= noisy_width:
            #             #if the destination pixel is outside the image, then we will bring it inside the image
            #             #by subtracting the distance it is outside the image from the image size
            #             distance_y_outside = dest_y - (noisy_width - 1)
            #             dest_y = noisy_width - distance_y_outside - 1
            ################# ENDS --- FOR WEIGHTS OUTSIDE BOUNDARIES   --------------##################
                    #now we have the location of the destination pixel in the noisy image
                    #we will read the patch from the padded reference image
                    #the patch is centered at the destination pixel in the noisy image
                    #so the patch center in the padded reference image is (dest_x+(patch_size-1)/2, dest_y+(patch_size-1)/2)
                    # patch_dest = padded_ref_image[\
                    #     dest_x+(patch_size-1)//2-(patch_size-1)//2:dest_x+(patch_size-1)//2+(patch_size-1)//2+1,\
                    #     dest_y+(patch_size-1)//2-(patch_size-1)//2:dest_y+(patch_size-1)//2+(patch_size-1)//2+1]
                    patch_dest = ndimage.interpolation.shift(reference_image, (-(dest_x-((patch_size-1)//2)), \
                                                                               -(dest_y-((patch_size-1)//2))),\
                                                                order=0, mode='reflect')[0:patch_size, 0:patch_size]
                    #now we have the current patch and the destination patch
                    #we will compute the distance between the 2 patches
                    #we will use the euclidean distance, and the distance is given by
                    # d = sqrt( sum( (patch_current - patch_dest)**2 ) )
                    #we will compute the distance between the 2 patches
                    # d = np.sum((patch_current - patch_dest)**2)
                    #compute l2 norm of the difference between the 2 patches
                    d = np.linalg.norm(patch_current - patch_dest)**2
                    #we normalize the distance by dividing by the patch size**2
                    d = d/(patch_size**2)
                    #now we have the distance between the 2 patches
                    #we will compute the weight between the 2 patches
                    #the weight is given by
                    # w = exp( -d**2 / (2*sigma**2) )
                    #get the hat function value at (i,j) and (i+dest_distance_x, j+dest_distance_y)
                    # hat = hat_function((i,j), (dest_x, dest_y), search_window)
                    #create a vector of differences between the current pixel and the destination pixel
                    vector = np.array([i-dest_x, j-dest_y])/(((search_window-1)//2)+1)
                    # hat = compute_hat(vector)
                    # numerator = np.exp(-d/(2*(h**2))) * compute_hat(vector)
                    numerator = np.exp(-d/(2*(h**2))) * compute_hat(vector)
                    # numerator *= hat
                    # #debug----------------
                    # print("i = ", i, "j = ", j)
                    # print("dest_x = ", dest_x, "dest_y = ", dest_y)
                    # #print source and target location
                    # print("source_location = ", source_location, "target_location = ", target_location)
                    # print("numerator = ", numerator)


                    # #---------------------debug
                    K[source_location,target_location] = numerator
                    # call hat function to get the other component of the weight
                    # the other component of the weight is given by the hat function
                    # hat  = hat_function((i,j), (dest_x, dest_y), search_window)
                    #now we have the weight between the 2 patches

    #    STEP : 12
    #we will normalize the weights
    #we wll construct a vector with the sum of all the rows of the kernel
    D = np.sum(K, axis=1)
    #we will raise the vector to the power of -0.5
    D = np.power(D, -0.5)
    #we will construct a diagonal matrix with the vector D
    D = np.diag(D)
    #now we will pre and post multiply the kernel with the diagonal matrix D
    K = np.dot(D, np.dot(K, D))
    #    STEP: 12 done

    #    STEP: 13 - 14
    #we will again create D as a vector of the sum of all the rows of the kernel
    D = np.sum(K, axis=1)
    #get the maximum value of D
    max_D = np.max(D)
    #we will calculate 1/max_D
    one_by_max_D = 1/max_D
    #we will multiply the kernel with 1/max_D
    K = K * one_by_max_D

    #   STEP: 13 - 14 done

    #    STEP: 15
    # to make it row/col stocastic, we will add the residue to the diagonal elements to make it row stochastic
    #calculate the residue
    #get the sum of all rows of the kernel
    D = np.sum(K, axis=1)
    #now construct a diagonal matrix with the vector D
    D = np.diag(D)
    K = K + np.eye(noisy_height * noisy_width) - D


    # #just to check what we are doing is correct, we will also compute the sum by looping over the window of 
    # # a pixel and summing the weights
    # #creta D as a matroix of zeros of the same size as num_pixels
    # D =  np.zeros((noisy_height * noisy_width, noisy_height * noisy_width))
    # #loop over the pixels in the noisy image
    # for i in range(noisy_height):
    #     for j in range(noisy_width):
    #         # we get the pixel location of current (i,j) pixel in the flattened image
    #         #thus tinstead of (i,j) we will use (i*noisy_width + j) to get the pixel location in the flattened image
    #         source_location = i*noisy_width + j

    #         sum = 0
    #         for dest_x in range( ( max(0, i-(search_window-1)//2) ), ( min(noisy_height, i+(search_window-1)//2+1) ) ):
    #             for dest_y in range( ( max(0, j-(search_window-1)//2) ), ( min(noisy_width, j+(search_window-1)//2+1) ) ):
    #                 #get the location of the destination pixel in the noisy image
    #         # for dest_x in range( ( max(0, i-(search_window)) ), ( min(noisy_height, i+(search_window)+1) ) ):
    #         #     for dest_y in range( ( max(0, j-(search_window)) ), ( min(noisy_width, j+(search_window)+1) ) ):
    #         #         #get the location of the destination pixel in the noisy image
    #                 target_location = dest_x*noisy_width + dest_y
    #                 #get the weight between the current pixel and the destination pixel
    #                 weight = K[source_location, target_location]
    #                 #add the weight to the sum
    #                 sum += weight
    #         #now we have the sum of the weights for the current pixel
    #         #we will add the sum to the diagonal matrix D
    #         D[source_location, source_location] = sum
    # #now we have the diagonal matrix D




    # return K, D
    return K