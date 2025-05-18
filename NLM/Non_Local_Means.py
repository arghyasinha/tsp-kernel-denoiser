
## imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy
from scipy import ndimage
from scipy import signal
from scipy import misc
from scipy import linalg
# from scipy.sparse.linalg import arpack
#we will import library to build DFT matrices
#we wont need fft but strictly DFT
#import scipy
import scipy
from scipy.linalg import dft
from decimal import Decimal


######################################################################################
######################################################################################
#------------------              DSG NLM FAST              ----------------------------
def integral_img_sq_diff(v,dx,dy):
    t = img2Dshift(v,dx,dy)
    diff = (v-t)**2
    #diff to double
    diff = diff.astype(np.float64)
    sd = np.cumsum(diff,axis=0)
    sd = np.cumsum(sd,axis=1)
    return(sd,diff,t)


def triangle(dx,dy,Ns):
    # r1 = np.abs(1 - np.abs(dx)/(Ns+1))
    #r1 is the positive difference between 1 and the absolute value of dx divided by Ns+1
    r1 = max(1 - (abs(dx)/(Ns+1)), 0)
    # r2 = np.abs(1 - np.abs(dy)/(Ns+1))
    #r2 is the positive difference between 1 and the absolute value of dy divided by Ns+1
    r2 = max(1 - (abs(dy)/(Ns+1)), 0)
    return r1*r2


def img2Dshift(v,dx,dy):
    row,col = v.shape
    t = np.zeros((row,col))
    typ = (1 if dx>0 else 0)*2 + (1 if dy>0 else 0)
    if(typ==0):
        t[-dx:,-dy:] = v[0:row+dx,0:col+dy]
    elif(typ==1):
        t[-dx:,0:col-dy] = v[0:row+dx,dy:]
    elif(typ==2):
        t[0:row-dx,-dy:] = v[dx:,0:col+dy]
    elif(typ==3):
        t[0:row-dx,0:col-dy] = v[dx:,dy:]
    return t


#thus the seperable trainagle filter takes in a 2D array and returns a 2D array
#the inputs are: dx, dy, Ns
#the output is: Hat
def seperable_trainagle(dx, dy, Ns, height, width):
    #calculate the value from dx,dy,Ns
    #the value is max(0, 1 - (abs(dx)/(Ns+1))) * max(0, 1 - (abs(dy)/(Ns+1)))
    value = max(0, 1 - (abs(dx)/(Ns+1))) * max(0, 1 - (abs(dy)/(Ns+1)))
    #we build the 2D array Hat with all values = value
    Hat = np.full((height, width), value).astype(np.float64)
    #but now all the pixel locations (i,j)  in the matrix Hat are source pixel and 
    #the destination ixel corresponding to each pixel location (i,j) is (i+dx, j+dy)
    #we have to put 0, in the matrix Hat, at the pixel locations (i,j) where the
    #destination pixel is outside the image
    #if dx is positive, then the destination pixel is outside the image if i+dx >= height
    #thus the last dx rows of the matrix Hat are all 0
    if dx >= 0:
        Hat[height-dx:,:] = 0
    #if dx is negative, then the destination pixel is outside the image if i+dx < 0
    #thus the first dx rows of the matrix Hat are all 0
    else:
        Hat[:abs(dx),:] = 0
    #if dy is positive, then the destination pixel is outside the image if j+dy >= width
    #thus the last dy columns of the matrix Hat are all 0
    if dy >= 0:
        Hat[:,width-dy:] = 0
    #if dy is negative, then the destination pixel is outside the image if j+dy < 0
    #thus the first dy columns of the matrix Hat are all 0
    else:
        Hat[:,:abs(dy)] = 0
    #return the matrix Hat converted to double
    return Hat.astype(np.float64)
#we write the main function to compute the DS_NLM algorithm
#we will implement the algorithm DSG_NLm written above of 4 steps
#we have 3 loops in the algorithm
#this algorithm is kind of loop unrolling


def DSG_NLM(input_image,guide_image,patch_rad,window_rad,sigma):
    #convert input_image to double
    input_image = input_image.astype(np.float64)
    guide_image = guide_image.astype(np.float64)

    if(len(input_image.shape) > 2):
        raise ValueError('Input must be a 2D array')
    height,width = input_image.shape
    u = np.zeros((height,width)).astype(np.float64)
    
    
    padded_guide = np.pad(guide_image,patch_rad,mode='symmetric').astype(np.float64)
    padded_v = np.pad(input_image,window_rad,mode='symmetric').astype(np.float64)
    # normalization_factor = (2*patch_rad*patch_rad*sigma*sigma)
    # normalization_factor = 2*(sigma*sigma)
    normalization_factor = (sigma*sigma)
    #convert to float64
    normalization_factor = np.float64(normalization_factor)

    # 0th loop
    W0 = np.zeros((height,width)).astype(np.float64)
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            # hat = triangle(dx,dy,window_rad)
            hat = seperable_trainagle(dx,dy,window_rad,height,width)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            # w = hat * np.exp(-sqdist1/(2*sigma**2))
            w = hat * np.exp(-sqdist1/normalization_factor).astype(np.float64)
            W0 = W0 + w

    # 1st loop
    W1 = np.zeros((height,width)).astype(np.float64)
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            # hat = triangle(dx,dy,window_rad)
            hat = seperable_trainagle(dx,dy,window_rad,height,width)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            # w = hat * np.exp(-sqdist1/(sigma**2))
            w = hat * np.exp(-sqdist1/normalization_factor).astype(np.float64)
            W0_pad = np.pad(W0,window_rad,mode='symmetric')
            W0_shift = img2Dshift(W0_pad,dx,dy)
            W0_temp = W0_shift[window_rad:window_rad+height,window_rad:window_rad+width]
            w1 = w / (np.sqrt(W0)*np.sqrt(W0_temp)).astype(np.float64)
            W1 = W1 + w1
    
    # 2nd loop
    alpha = 1/np.max(W1).astype(np.float64)
    W2 = np.zeros((height,width)).astype(np.float64)
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            if((dx != 0) or (dy != 0)):
                sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
                # hat = triangle(dx,dy,window_rad)
                hat = seperable_trainagle(dx,dy,window_rad,height,width)
                temp1 = img2Dshift(sd,patch_rad,patch_rad)
                temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
                temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
                temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
                res = temp1 + temp2 - temp3 - temp4
                sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
                # w = hat * np.exp(-sqdist1/(sigma**2))
                w = hat * np.exp(-sqdist1/normalization_factor).astype(np.float64)
                W0_pad = np.pad(W0,window_rad,mode='symmetric')
                W0_shift = img2Dshift(W0_pad,dx,dy)
                W0_temp = W0_shift[window_rad:window_rad+height,window_rad:window_rad+width]
                w2 = (alpha*w) / (np.sqrt(W0)*np.sqrt(W0_temp)).astype(np.float64)
                v = padded_v[window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
                u = u + w2*v
                W2 = W2 + w2
    
    u = u + (1-W2)*input_image
    #convert u to double
    u = u.astype(np.float64)
    return u        # u = Denoised image

######################################################################################
######################################################################################
#------------------              DSG NLM FAST ENDS         ----------------------------



def NLM(input_image,guide_image,patch_rad,window_rad,sigma, return_D_matrix=False):
    #convert input_image to double
    input_image = input_image.astype(np.float64)
    guide_image = guide_image.astype(np.float64)

    if(len(input_image.shape) > 2):
        raise ValueError('Input must be a 2D array')
    
    height,width = input_image.shape

    U = np.zeros((height, width)).astype(np.float64)  # To hold denoised image
    Z = np.zeros((height, width)).astype(np.float64)   # To hold accumulated weights
    
    padded_guide = np.pad(guide_image,patch_rad,mode='symmetric')
    padded_v = np.pad(input_image,window_rad,mode='symmetric')

    # normalization_factor = (2*patch_rad*patch_rad*sigma*sigma)
    normalization_factor = (sigma*sigma)
    #convert to float64
    normalization_factor = np.float64(normalization_factor)
    
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            # hat = triangle(dx,dy,window_rad)
            hat = seperable_trainagle(dx,dy,window_rad,height,width)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = hat * np.exp(-sqdist1/normalization_factor).astype(np.float64)
            v = padded_v[window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
            U = U + w*v
            Z = Z + w
    U = U/Z
    if return_D_matrix:
        return U , Z    # U = Denoised image, Z = Normalization coefficients
    return U        # U = Denoised image




#we will now write the adjoint operator of NLM: 

def NLM_adjoint(input_image, guide_image, patch_rad, window_rad, sigma, return_D_matrix=False, D_matrix=None):
    if D_matrix is None:
        # First call NLM to get D_matrix
        _, D_matrix = NLM(input_image = input_image, guide_image = guide_image, patch_rad = patch_rad, window_rad = window_rad, \
                          sigma = sigma, return_D_matrix = True)
    
    # Step 1: Apply D^{-1} to the input
    D_inv_input = input_image / D_matrix
    
    # Step 2: Apply NLM to the resultant
    NLM_result = NLM(D_inv_input, guide_image, patch_rad, window_rad, sigma)
    
    # Step 3: Apply D to the output
    NLM_adjoint_result = NLM_result * D_matrix
    
    if return_D_matrix:
        return NLM_adjoint_result, D_matrix
    return NLM_adjoint_result



