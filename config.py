import math
import PIL
from PIL import Image

from utilities import forward_models
from utilities.forward_models import *
importlib.reload(forward_models)


from NLM import Non_Local_Means
from NLM.Non_Local_Means import NLM, DSG_NLM
importlib.reload(Non_Local_Means)


from utilities import utils
from utilities.utils import *
importlib.reload(utils)



image_number = 2
# power_method_iterations = 5
power_method_iterations = 15
# power_method_iterations = 10
# plot_power_method = True
plot_power_method = False
verbose_power_method = False

#hyperparameters
eta = 0.9

#paths



def get_forward_model_args(application, image_shape, downscale=0.5, kernal_size=7, sigma=None, prob_observe=0.3):
    if application == 'inpainting':
        A_function = inpainting
        A_kwargs = {'prob_observe': prob_observe}
        A_function_adjoint = inpainting
        A_adjoint_kwargs = A_kwargs.copy()
    elif application == 'superresolution':
        A_function = superresolution
        if sigma is None:
            # if downscale == 0.5:
            #     sigma = sigma = math.sqrt(3.37)
            # else:           
            #     sigma = 2*(1/ (math.pi * downscale))
            local_sigma = 1.6

        A_kwargs = {'f_blur': gaussian_blurring, 'blur_kwargs': {'kernal_size': kernal_size, 'sigma': local_sigma}, 
                    'downsample_fraction_x': downscale, 'downsample_fraction_y': downscale}
        A_function_adjoint = superresolution_adjoint
        A_adjoint_kwargs = A_kwargs.copy()
        A_adjoint_kwargs['original_image_rows'] = image_shape[0]
        A_adjoint_kwargs['original_image_cols'] = image_shape[1]
    elif application == 'uniform_blurring':
        A_function = uniform_blurring
        A_kwargs = {'kernal_size': kernal_size}
        A_function_adjoint = uniform_blurring
        A_adjoint_kwargs = A_kwargs.copy()
    elif application == 'gaussian_blurring':
        A_function = gaussian_blurring
        # A_kwargs = {'kernal_size': kernal_size, 'sigma': sigma}
        A_kwargs = {'kernal_size': 12, 'sigma': 1.6}
        A_function_adjoint = gaussian_blurring
        A_adjoint_kwargs = A_kwargs.copy()
    else:
        raise ValueError(f"Application {application} is not implemented")

    return A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs


def get_image(image_number):
    image_path = 'images/Set12/'
    # image_path = 'images/MicroCrops/'
    image_path = get_path(image_path = image_path, number = image_number)
    #read the image
    image = Image.open(image_path)
    #debug----------------
    print('image.size: ', image.size)
    #print max min
    print_max_min(image, 'original image')
    #debug----------------
    #convert the image to a numpy array
    image = np.array(image)
    #to double
    image = image.astype(np.float64)
    #to range 0 to 1P_operator
    image = image/255.0
    return image


def get_denoiser(denoiser_name):
    if denoiser_name == 'DSG_NLM':
        denoiser = DSG_NLM
        denoiser_kwargs = {'patch_rad': 1, 'window_rad': 3, 'sigma': 100.0/255.0}
        # denoiser_kwargs = {'patch_rad': 3, 'window_rad': 10, 'sigma': 40.0/255.0}
        # denoiser_kwargs = {'patch_rad': 3, 'window_rad': 10, 'sigma': 100.0/255.0}
        # denoiser_kwargs = {'patch_rad': 3, 'window_rad': 10, 'sigma': 75.0/255.0}
    elif denoiser_name == 'NLM':
        denoiser = NLM
        denoiser_kwargs = {'patch_rad': 1, 'window_rad': 3, 'sigma': 100.0/255.0}
    else:
        raise ValueError(f"Denoiser {denoiser_name} is not implemented")

    return denoiser, denoiser_kwargs

