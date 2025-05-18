import numpy as np
from utilities.contractive_factor import power_method_for_images_non_symmetric
from config import *
from utilities.utils import D



#we will wrte a function to compute the linear operation P from teh linear operators W and G
#input: input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta
#output: output_image
def P_operator(input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta):
    intermediate_image = input_image - eta * A_function_adjoint(A_function(input_image, **A_kwargs), **A_adjoint_kwargs)
    output_image = denoiser(intermediate_image, **denoiser_kwargs)
    return output_image

#we will write a function to compute norm of P, if scaled = True then D norm, else 2 norm

def P_operator_adjoint(input_image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta):
    intermediate_image = denoiser(input_image, **denoiser_kwargs)
    output_image = intermediate_image - eta * A_function_adjoint(A_function(intermediate_image, **A_kwargs), **A_adjoint_kwargs)
    return output_image


#we will write a function to compute norm of P, if scaled = True then D norm, else 2 norm

def P_operator_norm(image, denoiser, denoiser_kwargs, A_function, A_kwargs, A_function_adjoint, A_adjoint_kwargs, eta, max_iterations= power_method_iterations, plot=plot_power_method, verbose = verbose_power_method ):
    if 'guide_image' not in denoiser_kwargs:
        denoiser_kwargs['guide_image'] = image
    # we will use the function : power_method_for_images_non_symmetric
    sigma, u, v = power_method_for_images_non_symmetric(functional=P_operator, functional_adjoint=P_operator_adjoint, \
                                                        image_height=image.shape[0], image_width=image.shape[1], \
                                                            args_dict_functional={'denoiser': denoiser, 'denoiser_kwargs': denoiser_kwargs, 'A_function': A_function, 'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, 'A_adjoint_kwargs': A_adjoint_kwargs, 'eta': eta}, \
                                                                args_dict_functional_adjoint={'denoiser': denoiser, 'denoiser_kwargs': denoiser_kwargs, 'A_function': A_function, 'A_kwargs': A_kwargs, 'A_function_adjoint': A_function_adjoint, 'A_adjoint_kwargs': A_adjoint_kwargs, 'eta': eta}, \
                                                                    max_iterations= max_iterations , plot=plot, verbose = verbose)
    return sigma

