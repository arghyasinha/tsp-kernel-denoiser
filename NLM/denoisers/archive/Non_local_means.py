## In this notebook, we will introduce the fast implementation of the non-local means algorithm.
#imports
import numpy as np

#import denoise_nl_means from skimage
from skimage.restoration import denoise_nl_means, estimate_sigma

def integral_img_sq_diff(v,dx,dy):
    t = img2Dshift(v,dx,dy)
    diff = (v-t)**2
    sd = np.cumsum(diff,axis=0)
    sd = np.cumsum(sd,axis=1)
    return(sd,diff,t)

 


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


def NLM(input_image,patch_rad,window_rad,sigma,guide_image=None):

    #if guide_image is not provided, use input_image as guide
    if(guide_image is None):
        guide_image = input_image

    if(len(input_image.shape) > 2):
        raise ValueError('Input must be a 2D array')
    
    height,width = input_image.shape

    U = np.zeros((height, width)) # To hold denoised image
    Z = np.zeros((height, width))  # To hold accumulated weights
    
    padded_guide = np.pad(guide_image,patch_rad,mode='symmetric')
    padded_v = np.pad(input_image,window_rad,mode='symmetric')

    normalization_factor = (2*patch_rad*patch_rad*sigma*sigma)
    # normalization_factor = (sigma*sigma)
    #convert to float64
    # normalization_factor = np.float64(normalization_factor)
    
    for dx in np.arange(-window_rad,window_rad+1):
        for dy in np.arange(-window_rad,window_rad+1):
            sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
            temp1 = img2Dshift(sd,patch_rad,patch_rad)
            temp2 = img2Dshift(sd,-patch_rad-1,-patch_rad-1)
            temp3 = img2Dshift(sd,-patch_rad-1,patch_rad)
            temp4 = img2Dshift(sd,patch_rad,-patch_rad-1)
            res = temp1 + temp2 - temp3 - temp4
            sqdist1 = res[patch_rad:patch_rad+height,patch_rad:patch_rad+width]
            w = np.exp(-sqdist1/normalization_factor)
            v = padded_v[window_rad+dx:window_rad+dx+height,window_rad+dy:window_rad+dy+width]
            U = U + w*v
            Z = Z + w
    U = U/Z
    return U     # U = Denoised image, Z = Normalization coefficients