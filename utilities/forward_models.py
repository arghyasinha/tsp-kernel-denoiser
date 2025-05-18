 
#imports
#import utils.py from parent directory
import sys
sys.path.append('../')

import importlib
#import utils
# import utils
# importlib.reload(utils)
# from utils import *

#import libraries for gaussian blurring on numpy arrays
import scipy.ndimage as ndimage
import scipy.signal as signal
from skimage.util import random_noise
from skimage import data, img_as_float
#for convolution on numpy arrays
import numpy as np
import time
#import #we will use scipy.ndimage.gaussian_filter for gaussian blurring
from scipy.ndimage import gaussian_filter
#import kernal
from scipy.signal import convolve2d
#gaussian kernal
from scipy.ndimage.filters import gaussian_filter
#we will write a function that will take an image and parameters for gaussian blurring and return a blurred image
#input: image, kernal size, sigma
#output: blurred image
#processing: we want output image to be of same size as input image, so use symmetric padding

#we will write a function that will take an image and parameters for gaussian blurring and return a blurred image
#input: image, kernal size, sigma
#output: blurred image
#processing: we want output image to be of same size as input image, so use symmetric padding

current_gaussian_kernal = None
current_sigma = None
current_kernal_size = None


def gaussian_kernel(size, sigma):
    """Create a Gaussian kernel with the given size and sigma."""
    size = int(size) // 2
    x, y = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1))
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-(x**2 + y**2) / (2.0 * sigma**2)) * normal

    #normalize the kernal
    g = g / np.sum(g)
    
    return g


def gaussian_blurring(input_image, kernal_size ,sigma ):
    global current_gaussian_kernal
    global current_sigma
    global current_kernal_size
    #extract parameters
    # kernal_size, sigma
    # kernal_size = parameter_dict['kernal_size']
    # sigma = parameter_dict['sigma']
    #we will write vectorized fast code possibly using libraries
    #we will use scipy.ndimage.gaussian_filter
    #we will use symmetric paddin

    
    # output = gaussian_filter(input=image, sigma=sigma, mode='mirror', radius = kernal_size)
    output = gaussian_filter(input = input_image, sigma=sigma, mode='mirror')
    #we will constrcut a gaussian kernal and use it for convolution
    #we will use scipy.signal.convolve2d
    #we will use symmetric padding
    #if current_sigma and current_kernal_size != sigma and kernal_size:
    #we will call the gaussian kernal function to create a kernal
    #else we will use the current kernal saved in current_gaussian_kernal
    #use global keyword to access global variables
    
    
    # if current_sigma != sigma or current_kernal_size != kernal_size or current_gaussian_kernal is None:
    #     current_gaussian_kernal = gaussian_kernel(kernal_size, sigma)
    #     current_sigma = sigma
    #     current_kernal_size = kernal_size

    # # #convolve the kernal with the input image, 
    # # # output of same size, use symmetric padding
    # output = convolve2d(input_image, current_gaussian_kernal, mode='same')
    
    return output


#unifrom blurring
#we will write a function that will take an image and parameters for uniform blurring and return a blurred image
#input: image, kernal size
#output: blurred image
#processing: we want output image to be of same size as input image, so use symmetric padding

def uniform_blurring(input_image, kernal_size ):
    #extract parameters 
    # kernal_size
    #we will write vectorized fast code possibly using libraries
    #we will use scipy.ndimage.gaussian_filter
    #we will use symmetric padding
    normalization_factor = (kernal_size)**2

    #build the kernal with all values equal to: normalization_factor
    kernal = np.ones((kernal_size,kernal_size))/normalization_factor

    # output = gaussian_filter(input=image, sigma=sigma, mode='mirror', radius = kernal_size)
    output = signal.convolve2d(input_image, kernal, mode='same',\
                                boundary='symm')

    return output


#we will write a code for inpainting
#we will use 1 for the present pixels and 0 for the missing pixels
#parameters: input_image, probability of missing pixels, window_rad
#output: inpainted image 

#we will also have global variables for the current window_rad and current probability and current inpainting image
#we will us ethese global variables if the call to inpainting function is for the same window_rad and probability

current_inpainting_image = None
current_prob_observe = None

def inpainting(input_image, prob_observe, inpainting_matrix = None, return_matrix = False):
    #we will use global variables for the current window_rad and current probability and current inpainting image
    #we will us ethese global variables if the call to inpainting function is for the same window_rad and probability
    global current_inpainting_image
    global current_prob_observe

    #we will check if the current window_rad and current probability and current inpainting image are same as the input parameters
    #if they are same then we will use the current inpainting image
    #else we will create a new inpainting image
    #or current_inpainting_image.shape != input_image.shape
    if inpainting_matrix is not None:
        output = input_image*inpainting_matrix

        if return_matrix:
            return output, inpainting_matrix

    else:

        if current_prob_observe != prob_observe or current_inpainting_image is None\
            or current_inpainting_image.shape[0] != input_image.shape[0] or current_inpainting_image.shape[1] != \
                input_image.shape[1]:
            #we will create a new inpainting image
            #we will create a random image with values between 0 and 1
            random_image = np.random.rand(input_image.shape[0], input_image.shape[1])
            #we will create a mask image with 1 where the random image is less than prob_observe and 0 otherwise
            current_inpainting_image = random_image < prob_observe
            #we will update the global variables
            current_prob_observe = prob_observe

            # #we have to check that in the current_inpainting_image each window has atleast one pixel that is 1
            # #so we will loop over each window  in current_inpainting_image of size window_rad*window_rad
            # #so we will start our window from top left corner and with a stride of 1
            # #we will loop over the rows and same for columns
            # #we will start from row 0 and go till row input_image.shape[0] - window_rad
            # for row in range(window_rad, input_image.shape[0] - window_rad):
            #     #we will start from column 0 and go till column input_image.shape[1] - window_rad
            #     for col in range(window_rad, input_image.shape[1] - window_rad):
            #         #we will check if the current window has atleast one pixel that is 1
            #         #the window is 2*window_rad + 1, centered at (row, col)
            #         #thus find the sum of the current window
            #         window_sum = np.sum(current_inpainting_image[row - window_rad:row + window_rad + 1, \
            #                                                     col - window_rad:col + window_rad + 1])
            #         #if the window sum is 0 then we will set the current pixel to 1
            #         if window_sum == 0:
            #             current_inpainting_image[row, col] = 1

        #we do pointwise multiplication of the input image and the current_inpainting_image 
        output = input_image*current_inpainting_image

        if return_matrix:
            return output, current_inpainting_image
       
    return output
    
#we will write a function to implement superresolution
#superresolution is  as an operation A = SB, were S is sampling and B is blurring
#we will implement as operation, we dont construct the matrix

#so A^T.A = B^T.S^T.S.B, but S^T.S is just inpainting, with some pixels not sampled as black
#so A^T.A = B^T.[Inpainting].B
#thus we will write directly the function  for the A^T.A operation for superresolution
# #using already defined functions for inpainting and gaussian blur

#input: input_image, f_B, f_B_adj, arg_dict_B, f_A, arg_dict_A, 
#output: output_image
#processing: we will apply the f_B to the input_image with arg_dict_B
#then to the output image apply f_A with arg_dict_A
#then apply f_B_adj to the output image with arg_dict_B
#then return the output image

# def superresolution(input_image, f_B, f_B_adj, arg_dict_B, f_A, arg_dict_A):
#     #we will apply the f_B to the input_image with arg_dict_B
#     output_image = f_B(input_image, **arg_dict_B)
#     #then to the output image apply f_A with arg_dict_A
#     output_image = f_A(output_image, **arg_dict_A)
#     #then apply f_B_adj to the output image with arg_dict_B
#     output_image = f_B_adj(output_image, **arg_dict_B)
#     #then return the output image
#     return output_image
 
 #  superresolution----------------------------------------------
 #we will have global variables, to reuse the same downsampling matrix until the input image size changes or 
#until the downsample_fraction changes
current_downsampling_matrix = None
current_downsampling_fraction = None
current_input_image_size = None
current_output_image_size = None

#write a function to construct Downsample matrix
#input: original_image_rows,original_image_cols, ratio_x, ratio_y
def construct_downsample_matrix( original_image_rows, original_image_cols, downsampled_image_rows,\
    downsampled_image_cols,  ratio_x, ratio_y , verbose= False):
    #define global variables
    global current_downsampling_matrix
    global current_downsampling_fraction
    global current_input_image_size
    global current_output_image_size
    #----- ----- ---- ----- 
    #now we will construct the downsampling matrix
    original_image_pixels = original_image_rows * original_image_cols
    downsampled_image_pixels = downsampled_image_rows * downsampled_image_cols
    #let us check if we need to construct a new downsampling matrix, then only we will construct it
    #check if any global variables are None
    
    if current_downsampling_matrix is None or  current_downsampling_fraction is None or \
        current_input_image_size is None or current_output_image_size is None or \
            current_input_image_size != (original_image_rows, original_image_cols) or \
        current_downsampling_fraction != (ratio_x, ratio_y)\
        or current_output_image_size != (downsampled_image_rows, downsampled_image_cols):
    # if True:
        # #debug--
        # #print original rows, cols
        # print("original_image_rows", original_image_rows, "original_image_cols", original_image_cols)
        # #print downsampled row and column
        # print("downsampled_image_rows", downsampled_image_rows, "downsampled_image_cols", downsampled_image_cols)
        # #print ratio_x and ratio_y
        # print("ratio_x", ratio_x, "ratio_y", ratio_y)
        # #debug--
        #create a downsampling matrix of zeroes of size downsampled_image_pixels x original_image_pixels
        D = np.zeros((downsampled_image_pixels, original_image_pixels))
        #now we will construct the downsampling matrix by using foe loop by looping 
        #we will have a monotonic counter for the first coordinate i.e. teh row location
        #as we want a 1 in each row sequentially, so we would have a counter for the row and no. of 1s = downsampled_image_cols
        #but the column locations of 1s, we will do row by row of original image
        #i.e. we will start with the 0th row and then put 1s at a step of scale in the 0th row
        #when we exhaust the 0th row, we will go to  0+step row and then put 1s at a step of scale in the 0+step row
        #and so on..
        #now we will generate arrays of indices for the original image, x and y coordinates
        #based on these coordinates, we will calculate the row and column location in the downsampled image
        #and then we will put 1 at that location
        #if 1/ratio_x is an integer, then we will create the array of x coordinates in the original image
        #with a step size of 1/ratio_x
        if (1/ratio_x).is_integer():
            # #debug--
            # print("1/ratio_x", 1/ratio_x)
            # #debug--
            scale_x = int(1/ratio_x)
            x_coordinates = np.arange(0, original_image_rows, scale_x)[0:downsampled_image_rows] 
            # #debug--
            # print("x_coordinates", x_coordinates)
            # #debug--
        else:
            #now the ratio_x is not an integer, so we will create an array of indices of the original image
            #from 0 to original_image_rows - 1
            p_x = np.arange(0, original_image_rows)
            # #debug--
            # print("p_x", p_x)
            # #debug--
            #now randomly sample "downsampled_image_rows" elements from this array p_x
            x_coordinates = np.random.choice(p_x, downsampled_image_rows, replace = False)
            # #debug--
            # print("x_coordinates", x_coordinates)
            # #debug--
            #now sort the x_coordinates
            x_coordinates.sort()
            # #debug--
            # print("x_coordinates", x_coordinates)
            # #debug--

        #if 1/ratio_y is an integer, then we will create the array of y coordinates in the original image
        #with a step size of 1/ratio_y
        if (1/ratio_y).is_integer():
            # #debug--
            # print("1/ratio_y", 1/ratio_y)
            # #debug--
            scale_y = int(1/ratio_y)
            y_coordinates = np.arange(0, original_image_cols, scale_y)[0:downsampled_image_cols]
            # #debug--
            # print("y_coordinates", y_coordinates)
            # #debug--
        else:
            #now the ratio_y is not an integer, so we will create an array of indices of the original image
            #from 0 to original_image_cols - 1
            p_y = np.arange(0, original_image_cols)
            # #debug--
            # print("p_y", p_y)
            # #debug--
            #now randomly sample "downsampled_image_cols" elements from this array p_y
            y_coordinates = np.random.choice(p_y, downsampled_image_cols, replace = False)
            # #debug--
            # print("y_coordinates", y_coordinates)
            # #debug--
            #now sort the y_coordinates
            y_coordinates.sort()
            # #debug--
            # print("y_coordinates sorted", y_coordinates)
            # #debug--


        #now we have two arrays , one with the x coordinates of the original image, that would be sample
        #one with y coordinates of the original image, that would be sampled
        #we will generate teh coordinates of downsampled matrix by using these coordinates

        row_location = 0
        #for the row indices in x_coordinates
        for x in x_coordinates:
            #for the column indices in y_coordinates
            for y in y_coordinates:
                #we will put 1 at the row_location, col_location
                D[row_location, int(x * original_image_cols + y)] = 1
                #now we will increment the row_location by 1
                row_location += 1
                
        #we will update the global variables
        current_downsampling_matrix = D
        current_downsampling_fraction = (ratio_x, ratio_y)
        current_input_image_size = (original_image_rows, original_image_cols)
        current_output_image_size = (downsampled_image_rows, downsampled_image_cols)

    return current_downsampling_matrix
#we will write a function that is a downsampling operator
#input: image, image downsample_fraction_x, downsample_fraction_y
def downsamping_operator(image, downsample_fraction_x, downsample_fraction_y):
    #we will get the scale from the downsample fractions
    scale_x = 1/downsample_fraction_x
    scale_y = 1/downsample_fraction_y
    #scale should be an integer, if yes then convert to int, else error
    if scale_x.is_integer() and scale_y.is_integer():
        scale_x = int(scale_x)
        scale_y = int(scale_y)
    else:
        assert False, "scale should be an integer"
    #get the size of the image
    original_image_rows , original_image_cols = image.shape
    #get the size of the downsampled image, original rows and columns should be divisible by scale
    #else error
    if original_image_rows % scale_x == 0 and original_image_cols % scale_y == 0:
        downsampled_image_rows = original_image_rows // scale_x
        downsampled_image_cols = original_image_cols // scale_y
    else:
        assert False, "original image rows and columns should be divisible by scale"
    #create a downsampled image of zeros
    downsampled_image = np.zeros((downsampled_image_rows, downsampled_image_cols))
    #we will sample through the image and only pick every scale pixel to put in downsampled image
    #we can do this  in vectorized way
    #we will sample every scale index of the image rows starting from 0
    downsampled_indices_x = np.arange(0, original_image_rows, scale_x)
    #we will sample every scale index of the image columns starting from 0
    downsampled_indices_y = np.arange(0, original_image_cols, scale_y)
    #now we will select the subarray from teh image, where rows are selected by downsampled_indices_x
    #and columns are selected by downsampled_indices_y
    downsampled_image = image[downsampled_indices_x[:, None], downsampled_indices_y]
    #return the downsampled image
    return downsampled_image
#we will write an upsample operator, which will create a bigger image of size scale_x * original_image_rows,
#scale_y * original_image_cols, and will fill the new image with the values from the original image
#whereever we have values in the original image, and will fill the rest with zeros
#input: image, image downsample_fraction_x, downsample_fraction_y, original_image_rows, original_image_cols
def upsampling_operator(image, downsample_fraction_x, downsample_fraction_y, original_image_rows, \
                        original_image_cols):
    #we will get the scale from the downsample fractions
    scale_x = 1/downsample_fraction_x
    scale_y = 1/downsample_fraction_y
    #scale should be an integer, if yes then convert to int, else error
    if scale_x.is_integer() and scale_y.is_integer():
        scale_x = int(scale_x)
        scale_y = int(scale_y)
    else:
        #assert error
        assert False, "scale should be an integer"
        
    #get the size of the image
    downsampled_image_rows , downsampled_image_cols = image.shape
    #assert that the original image rows and columns are equal to the upsampled image rows and columns
    assert original_image_rows == downsampled_image_rows * scale_x and original_image_cols == \
        downsampled_image_cols * scale_y,  \
    "original image rows and columns should be equal to the upsampled image rows and columns"
    #create a upsampled image of zeros
    upsampled_image = np.zeros((original_image_rows, original_image_cols))
    #we will sample only the indices of the upsampled image that are divisible by scale
    #and fill the upsampled image with the values from the downsampled image
    #we will get the row indices of the upsampled image that are divisible by scale
    upsampled_indices_x = np.arange(0, original_image_rows, scale_x)
    #we will get the column indices of the upsampled image that are divisible by scale
    upsampled_indices_y = np.arange(0, original_image_cols, scale_y)
    #now we will select the subarray from teh upsampled image, where rows are selected by upsampled_indices_x
    #and columns are selected by upsampled_indices_y
    upsampled_image[upsampled_indices_x[:, None], upsampled_indices_y] = image
    #return the upsampled image
    return upsampled_image

#now we will write a function to create a mask function for superrresolution function
#the mask wuld means 0s at the places we dont have sampled from original image when we downsample and 
#upsample
#input: original_image_rows, original_image_cols, downsample_fraction_x, downsample_fraction_y
def create_superresolution_mask(original_image_rows, original_image_cols, downsample_fraction_x, \
                                 downsample_fraction_y):
    #we will get the scale from the downsample fractions
    scale_x = 1/downsample_fraction_x
    scale_y = 1/downsample_fraction_y
    #scale should be an integer, if yes then convert to int, else error
    if scale_x.is_integer() and scale_y.is_integer():
        scale_x = int(scale_x)
        scale_y = int(scale_y)
    else:
        assert False, "scale should be an integer"
    #scale should be dividing the original image rows and columns
    assert original_image_rows % scale_x == 0 and original_image_cols % scale_y == 0, \
    "scale should be dividing the original image rows and columns"
    #create a matrix of original image rows and columns with zeros
    mask = np.zeros((original_image_rows, original_image_cols))
    #we will sample only the indices of the upsampled image that are divisible by scale
    #and fill the upsampled image with the values from the downsampled image
    #we will get the row indices of the upsampled image that are divisible by scale
    upsampled_indices_x = np.arange(0, original_image_rows, scale_x)
    #we will get the column indices of the upsampled image that are divisible by scale
    upsampled_indices_y = np.arange(0, original_image_cols, scale_y)
    #now we will select the subarray from teh upsampled image, where rows are selected by upsampled_indices_x
    #and columns are selected by upsampled_indices_y
    mask[upsampled_indices_x[:, None], upsampled_indices_y] = 1
    #return the mask
    return mask
    
# we will write a function for sampling operation
#this will be a two step process: 
#1. apply a blur operation, for that we will pass the function f_blur, blur_kwargs
#2. apply a downsampling operation, for that we will pass the downsample_fraction
#downsample_fraction would be a float between 0 and 1, and we multiply the number of pixels of input image 
#by this fraction to get the number of pixels in the output image

def superresolution(input_image, f_blur, blur_kwargs, downsample_fraction_x , downsample_fraction_y ):  
    #image should be greayscale
    assert len(input_image.shape) == 2, "image is not grayscale"
    #----- BLUR OPERATION ------
    #apply the blur operation
    blurred_image = f_blur(input_image, **blur_kwargs)
    #----- DOWNSAMPLING OPERATION ------
    downsampled_image = downsamping_operator(blurred_image, downsample_fraction_x, \
                                             downsample_fraction_y)
    #return the downsampled image
    return downsampled_image


#we will define a function for transpose fo superresolution
#i.e. we will first apply Downsampling transpose operation, i.e. upsampling
#and then apply blur transpose operation, i.e blur as filters are symmetric
def superresolution_adjoint(downsampled_image, f_blur, blur_kwargs, original_image_rows, original_image_cols,\
     downsample_fraction_x , downsample_fraction_y):
    #image should be greayscale
    assert len(downsampled_image.shape) == 2, "image is not grayscale"
    #--------------------------------------------------------------------------
    blurred_image = upsampling_operator(downsampled_image, downsample_fraction_x , downsample_fraction_y,\
                                        original_image_rows, original_image_cols)
    #--------------------------------------------------------------------------
    #----- BLUR TRANSPOSE OPERATION ------
    #apply the blur transpose operation
    input_image = f_blur(blurred_image, **blur_kwargs)
    return input_image

#------------------------- SUPERRESOLUTION OPERATOR -------------------------
# #write a function to implement G function
# #input: input_image, fun_A , fun_A_adj , gamma, arg_dict
# #output: output_image

# #processing, we want the matrix product but we only have operators implemented as functions, so
# #we will do as in these matrix multiplications as operators acting on images

# def G(input_image, fun_A, fun_A_adj, gamma, arg_dict_A):
#     #we will call the function fun_A with input_image and arg_dict
#     #we will call the function fun_A_adj with the output of fun_A and arg_dict
#     #we will add the output of fun_A_adj to the input_image
#     #we will multiply the output of the above step by gamma
#     #we will return the output of the above step
#     output_image = input_image - gamma*fun_A_adj(fun_A(input_image, **arg_dict_A), **arg_dict_A)
#     return output_image


# #we will implement G function for superresolution

# #input: input_image, fun_A_adj_A, gamma arg_dict_A
# #output: output_image
# #processing: we will apply the fun_A_adj_A to the input_image with arg_dict_A
# #then to the output image multiply by gamma
# #then subtract the multiplied output from the input image
# #then return the output image
# def G_A_adj_A(input_image, fun_A_adj_A, gamma, arg_dict_A):
#     #we will apply the fun_A_adj_A to the input_image with arg_dict_A
#     output_image = fun_A_adj_A(input_image, **arg_dict_A)
#     #then to the output image multiply by gamma
#     output_image = gamma*output_image
#     #then subtract the multiplied output from the input image
#     output_image = input_image - output_image
#     #then return the output image
#     return output_image


# #we will wrte a function to compute the linear operation P from teh linear operators W and G
# #input: input_image, f_W, f_G, arg_dict_W, fun_A, fun_A_adj, gamma, arg_dict_A
# #output: output_image
# #processing: we habve to operate all the linear operatos on the input image
# #from right to left, i.e first G and then on the output of G, W
# #for g, we will call the function f_G with input_image, fun_A, fun_A_adj, gamma, arg_dict_A
# #for w, we will call the function f_W with the output of f_G, arg_dict_W

# def P(input_image, f_W, f_G, arg_dict_W, arg_dict_G):
#     #call f_G
#     output_image = f_G(input_image=input_image, **arg_dict_G)
#     #the parameter 'guide_image' has to be based on the output of f_G
#     #thus has to be dynamic and changed on each call to f_G
#     #lets change the value of arg_dict_W['guide_image'] to the output of f_G
#     # arg_dict_W['guide_image'] = output_image
#     arg_dict_W['guide_image'] = output_image
#     #call f_W
#     output_image = f_W(output_image, **arg_dict_W)
#     return output_image



# #we will write a function D to compute the linear operator D
# #the linear operation is just a multiplication of the input image by a matrix of the same size as image
# #and point wise multiplication
# #input: input_image, D_matrix
# #output: output_image
# #processing: D_matrix * input_image
# def D(input_image, D_matrix, D_power):
#     #we will raise each element of D_matrix to the power D_power
#     D_matrix = np.power(D_matrix, D_power)
#     #multiply the input image by D_matrix
#     output_image = D_matrix*input_image
#     return output_image
# #we will define P for D norm calculation, i.e. we need to build P as 
# # P = D^{-1/2} * W * G * D^{-1/2}
# #where each of them are linear operators

# #input would be same as the P function above with the addition of D_matrix
# #output: output_image
# #processing: 
# #1. we will the D function with input_image and D_matrix^{-1/2}
# #so teh arguments to D would be input_image, D_matrix, -1/2
# #the output from that would be the input to P
# #2. we will call P with the output of D and the rest of the arguments
# #3. we will call D with the output of P and D_matrix , 1/2
# #4. we will return the output of D
# def D_half_P_D_half_inv(input_image, f_W, f_G, arg_dict_W, arg_dict_G, D_matrix):
#     #call D
#     output_image = D(input_image=input_image, D_matrix=D_matrix, D_power=-1/2)
#     #call P
#     output_image = P(input_image=output_image, f_W=f_W, f_G=f_G, arg_dict_W=arg_dict_W, arg_dict_G=arg_dict_G)
#     #call D
#     output_image = D(input_image=output_image, D_matrix=D_matrix, D_power=1/2)
#     return output_image


# ###############################   ADMM   ########################################
# #################################################################################

# #we will implement operators for ADMM
# #we will implement th eoperators J and R
# #at the end, we will be interested in plotting the norm of R wrt rho


# # $J := (2W - I)(2(I+\rho A^T A) - I) $

# # $R := \frac{1}{2} (I + J)$

# # A is inpainting operation, so A^T = A
# # W is the NLM operation
# #we will implement first J's 2nd part and then J's 1st part
# # implement the part of J that is not dependent on rho
# admm_current_inpainting_image = None
# admm_current_prob_observe = None
# #input: input_image, rho, f_A
# def f_F(input_image, rho , prob_observe ):
#     #we will use global variables for the current window_rad and current probability and current inpainting image
#     #we will us ethese global variables if the call to inpainting function is for the same window_rad and probability
#     global admm_current_inpainting_image
#     global admm_current_prob_observe
#     #create a matrix of the same size as input_image
#     all_ones = np.ones(input_image.shape)

#     if admm_current_prob_observe != prob_observe or admm_current_inpainting_image is None\
#         or admm_current_inpainting_image.shape[0] != input_image.shape[0] or \
#             admm_current_inpainting_image.shape[1] != input_image.shape[1]:

#             #we will create a new inpainting image
#         #we will create a random image with values between 0 and 1
#         random_image = np.random.rand(input_image.shape[0], input_image.shape[1])
#         #we will create a mask image with 1 where the random image is less than prob_observe and 0 otherwise
#         admm_current_inpainting_image = random_image < prob_observe
#         #we will update the global variables
#         admm_current_prob_observe = prob_observe
        
#     A = admm_current_inpainting_image
#     #we will create a matrix all_ones + rho * A
#     matrix = all_ones + rho * A
#     #raise each element of the matrix to the power of -1
#     matrix = np.power(matrix, -1)
#     #the output would be the element wise multiplication of the input image and the matrix multiplied by 2
#     #then subtracted by the input image
#     output = 2 * (matrix * input_image ) - input_image
#     return output

    
# #we will implement the part of J that is dependent on W, the pre part
# #it will call the J_post_part function
# #and one more parameter, the f_W function
# #input: input_image, rho, f_W, arg_dict_W, prob_observe = 0.3
# #output: output 
# #processing: first we will call J_post_part to operate on the input_image, and get the output
# #on the output, we want to do 2W-I
# #i.e. we will call the function f_W with the output as input and arg_dict_W as arg_dict, them multiply
# #new output by 2 and subtract theold output

# def J(input_image, f_W, arg_dict_W, rho, prob_observe ):
#     #first we will call J_post_part to operate on the input_image, and get the output
#     output = f_F(input_image, rho, prob_observe)
#     #on the output, we want to do 2W-I
#     #i.e. we will call the function f_W with the output as input and arg_dict_W as arg_dict, them multiply
#     #new output by 2 and subtract theold output

#     output = 2 * f_W(output, **arg_dict_W) - output
#     return output
# #we will implement R
# #we will call J , then we will add the input_image to the output of J and 
# #then divide the whole by 2
# #input: input_image, f_W, arg_dict_W, rho, prob_observe = 0.3
# #output: output
# #processing: first we will call J to operate on the input_image, and get the output
# #then we will add the input_image to the output of J and then divide the whole by 2

# def R(input_image, f_W, arg_dict_W, rho, prob_observe ):
#     #first we will call J to operate on the input_image, and get the output
#     output = J(input_image, f_W, arg_dict_W, rho, prob_observe)
#     #then we will add the input_image to the output of J and then divide the whole by 2
#     output = (input_image + output) / 2
#     return output

# #we will implement a function that returns D^{1/2} R D^{- 1/2}
# #this would be used for norm calculating function to calculate D norm
# #input: input_image, f_W, arg_dict_W, rho, prob_observe , D_matrix
# #output: output
# #processing: 
# # 1. first we will the D function with input_image and D_matrix^{-1/2}
# #so teh arguments to D would be input_image, D_matrix, -1/2
# #the output from that would be the input to R 
# #2. we will call R with the output of D and the rest of the arguments
# #3. we will call D with the output of P and D_matrix , 1/2
# #4. we will return the output of D
# def D_half_R_D_half_inv(input_image, f_W, arg_dict_W, rho, prob_observe , D_matrix):
#     #first we will the D function with input_image and D_matrix^{-1/2}
#     #so teh arguments to D would be input_image, D_matrix, -1/2
#     #the output from that would be the input to R 
#     output = D(input_image, D_matrix, -1/2)
#     #we will call R with the output of D and the rest of the arguments
#     output = R(output, f_W, arg_dict_W, rho, prob_observe)
#     #we will call D with the output of P and D_matrix , 1/2
#     output = D(output, D_matrix, 1/2)
#     #we will return the output of D
#     return output