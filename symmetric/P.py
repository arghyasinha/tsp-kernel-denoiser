import numpy as np
from PIL import Image
import math

import sys
sys.path.append('..')

from config import *
from operators_ISTA import P_operator_norm
from utilities.contractive_factor import second_largest_eigenvalue, power_method_for_images_non_symmetric

application = 'inpainting'




def theorem_5_symm( application = 'inpainting', application_kwargs = {}, denoiser_kwargs = {}, eta = 0.1, image_number = image_number, image=None):
    # Step 1: Get the image from image_number if image is None
    if image is None:
        image = get_image(image_number)

    # Step 2: Build arguments for A_operator, A_operator_adjoint
    image_shape = image.shape
    A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs = get_forward_model_args(application = application, image_shape = image_shape)
    #we will overwrite the keyword arguments to A_kwargs and A_adjoint_kwargs with application_kwargs
    for key, value in application_kwargs.items():
        if key in A_kwargs:
            A_kwargs[key] = value
        if key in A_adjoint_kwargs:
            A_adjoint_kwargs[key] = value   

    # Step 3: Use DSG_NLM as denoiser
    denoiser , denoiser_kwargs = get_denoiser(denoiser_name='DSG_NLM')
    for key, value in denoiser_kwargs.items():
        if key in denoiser_kwargs:
            denoiser_kwargs[key] = value
    #add image to keyword arguments of denoiser. 'guide_image'
    denoiser_kwargs['guide_image'] = image

    # Step 4: Compute lhs as the P_operator_norm
    lhs = P_operator_norm(image=image, denoiser=denoiser, denoiser_kwargs=denoiser_kwargs, A_function=A_function, A_kwargs=A_kwargs, A_function_adjoint=A_function_adjoint, A_adjoint_kwargs=A_adjoint_kwargs, eta=eta)
    lhs = lhs * lhs

    # Step 5: Compute the second largest eigenvalue of W (denoiser)
    lambda_2, _ = second_largest_eigenvalue(f=denoiser, input_image=image, args_dict=denoiser_kwargs, max_iterations=power_method_iterations, plot=plot_power_method, verbose=verbose_power_method)
    # Step 6: Compute mu from the application keyword arguments
    if application == 'deblurring':
        mu = 1.0  # Default value of mu is 1.0
        rhs = lambda_2 ** 2 + (1 - lambda_2 ** 2) * (1 - eta) ** 2
    elif application == 'inpainting' or application == 'superresolution':
        mu = A_kwargs['prob_observe'] if application == 'inpainting' else A_kwargs['downsample_fraction_x']
        rhs = lambda_2 ** 2 + (1 - lambda_2 ** 2) * (1 - eta * (2 - eta) * mu)


    # Step 7: Verify the inequality
    inequality_holds = lhs<= rhs

    return {
        'P_operator_norm': lhs,
        'lambda_2': lambda_2,
        'rhs': rhs,
        'inequality_holds': inequality_holds
    }

