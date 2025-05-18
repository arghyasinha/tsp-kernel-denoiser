import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

from symmetric.P import theorem_5_symm

from utilities.utils import get_image_name, generate_random_image_with_noise
from config import get_image

algorithm = 'symmetric'

application = 'inpainting'
application_kwargs = {}

#hyperparameters
image_number = 5 #butterfly
eta = 0.9

# sweep-hyperparameters
mu_inpainting_sweep_size = 5
noise_sigma_sweep_size = 50
simga_min = 1e-3
sigma_max = 1e+3
sigma_sweep_size = 50

if algorithm == 'symmetric':    
    output_path = 'output/thm_5_symm'
    theorem_5 = theorem_5_symm
    norm_label = r'$\|P\|_2^2$'

else:
    raise ValueError(f"Algorithm {algorithm} not supported")

os.makedirs(output_path, exist_ok=True)

def analyze_eta_range(image_number=None, image=None, image_name = None, eta_min=0.0, eta_max=2.0, num_samples=50, application=application, application_kwargs=application_kwargs):
    eta_values = np.linspace(eta_min, eta_max, num_samples)
    results = []

    for eta in eta_values:
        result = theorem_5(application=application, application_kwargs=application_kwargs, eta=eta, image_number=image_number, image=image)
        results.append((eta, result['P_operator_norm'], result['lambda_2'], result['rhs'], result['inequality_holds']))
        print(f"eta: {eta}, P_operator_norm: {result['P_operator_norm']}, lambda_2: {result['lambda_2']}, lambda_2^2: {np.power(result['lambda_2'], 2)}, rhs: {result['rhs']}, inequality_holds: {result['inequality_holds']}")


    

    application_kwargs_str = "_".join([f"{key}_{value}" for key, value in application_kwargs.items()])
    image_name_string = f"image_{image_number if image is None else image_name}"
    filename = f"{output_path}/eta_range_{application}{application_kwargs_str}_{image_name_string}"
    np.save(f"{filename}.npy", results)

    # Plotting the results
    eta_values, P_operator_norm_values, lambda_2_values, rhs_values, _ = zip(*results)
    plt.plot(eta_values, P_operator_norm_values, label=norm_label)
    plt.plot(eta_values, lambda_2_values, label=r'$\lambda_2$')
    #plot lambda_2^2
    plt.plot(eta_values, np.power(lambda_2_values, 2), label=r'$\lambda_2^2$')
    plt.plot(eta_values, rhs_values, label='bound')
    plt.xlabel('eta')
    # plt.ylabel('Values')
    plt.title(f'{application} - {"_".join([f"{key}_{value}" for key, value in application_kwargs.items()])} - {image_name_string}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename}.png")
    plt.show()
    
    return results, eta_values
    


    

def analyze_application_kwargs(image_number=None, image=None, image_name=None, eta=eta, application=application, application_kwargs=application_kwargs, application_kwargs_list=[], kwarg=None):
    if application == 'inpainting':
        if kwarg is None:
            kwarg = 'prob_observe'
            if not application_kwargs_list: 
                application_kwargs_list = np.linspace(0.0, 1.0, mu_inpainting_sweep_size)
    else:
        raise ValueError(f"Application {application} not supported")

    results = []

    for app_kwargs in application_kwargs_list:
        if kwarg == 'downsample_fraction_x':
            application_kwargs = {'downsample_fraction_x': app_kwargs, 'downsample_fraction_y': app_kwargs}
        else:
            application_kwargs = {kwarg: app_kwargs}
        result = theorem_5(application=application, application_kwargs=application_kwargs, eta=eta, image_number=image_number, image=image)
        results.append((app_kwargs, result['P_operator_norm'], result['rhs'], result['lambda_2'], result['inequality_holds']))
        print(f"{kwarg}: {app_kwargs}, P_operator_norm: {result['P_operator_norm']}, rhs: {result['rhs']}, lambda_2: {result['lambda_2']}, lambda_2^2: {np.power(result['lambda_2'], 2)}, inequality_holds: {result['inequality_holds']}")

    application_kwargs_str = "_".join([f"{key}_{value}" for key, value in application_kwargs.items()])
    image_name_string = f"image_{image_number if image is None else image_name}"
    application_kwargs_range_str = str(len(application_kwargs_list))
    filename = f"{output_path}/{application}_{kwarg}_range_{application_kwargs_range_str}_{application_kwargs_str}_{image_name_string}_eta_{eta}.npy"
    np.save(filename, results)

    # Plotting the results
    app_kwargs_values, P_operator_norm_values, rhs_values, lambda_2_values, _ = zip(*results)
    plt.plot(app_kwargs_values, P_operator_norm_values, label=norm_label)
    plt.plot(app_kwargs_values, rhs_values, label='bound')
    plt.plot(app_kwargs_values, np.power(lambda_2_values, 2), label=r'$\lambda_2^2$')
    plt.xlabel('kwarg')
    plt.title(f'{application} - {"_".join([f"{key}_{value}" for key, value in application_kwargs.items()])} - {image_name_string}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename}.png")
    plt.show()


if __name__ == "__main__":

    image_name = get_image_name(image_number)
    image = get_image(image_number)

    #1. analyze_eta_range
    # analyze_eta_range(image= image, image_name=image_name)

    #2. analyze_application_kwargs
    analyze_application_kwargs(image= image, image_name=image_name, application_kwargs_list=[0.1,0.5,0.9])
    
