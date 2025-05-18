
import numpy as np
import math
import matplotlib.pyplot as plt
# Contraction - power method
# P
# $
# P = W G
# $
### the G operator is :

# $ 
# I_n - \gamma A^T A
# $



########################################    POWER METHOD FOR SPECTRAL NORM   ####################
########################################                                    ######################


######################################################################################
######################################################################################
#------------------              POWER METHOD              ----------------------------

#we will plot the gain in norm of each step, the euclidean distance between the two vectors, and the dot product
#for each step
#the graph has to be very crips and presentable and self explanatory
def plot_power_method(norm_gain_list, euclidean_distance_list, dot_product_list, save=False):
    # Plot the norm gain data
    plt.scatter(range(len(norm_gain_list)), norm_gain_list, color='red', label='norm gain')
    plt.plot(range(len(norm_gain_list)), norm_gain_list, color='salmon', linewidth=1.5)

    # Plot the euclidean distance data
    plt.scatter(range(len(euclidean_distance_list)), euclidean_distance_list, color='blue', label='euclidean distance')
    plt.plot(range(len(euclidean_distance_list)), euclidean_distance_list, color='lightblue', linewidth=1.5)

    # Plot the dot product data
    plt.scatter(range(len(dot_product_list)), dot_product_list, color='green', label='cosine similarity')
    plt.plot(range(len(dot_product_list)), dot_product_list, color='lightgreen', linewidth=1.5)

    # Label the x-axis
    plt.xlabel('iteration')

    # Add a legend to the plot
    plt.legend(loc='center right')

    # Label the last iteration on the x-axis
    plt.annotate(str(len(norm_gain_list)-1), xy=(len(norm_gain_list)-1, 0), xycoords='data',
                xytext=(0,-30), textcoords='offset points',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=-0.2"),
                ha='center', va='bottom', fontsize=10, color='r',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
                )
    plt.gca().tick_params(axis='x', which='major', pad=15)

    # Increase tick frequency for y axis intervals were more values are present
    # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    # Plot the y-axis values for all 3 metrics at the last iteration
    last_iter = len(norm_gain_list) - 1
    y_ticks = [norm_gain_list[last_iter], euclidean_distance_list[last_iter], dot_product_list[last_iter]]
    x_ticks = [last_iter] * len(y_ticks)
    # #print only 3 decimal places
    plt.annotate(str(round(norm_gain_list[last_iter], 3)), xy=(last_iter, norm_gain_list[last_iter]), xycoords='data',
                xytext=(0, -15), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, color='salmon',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
                )
    plt.annotate(str(round(dot_product_list[last_iter], 3)), xy=(last_iter, dot_product_list[last_iter]), xycoords='data',
                xytext=(0, -45), textcoords='offset points',  
                ha='center', va='bottom', fontsize=10, color='lightgreen',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
                )
    #for euclidean distance print the value above the point with distance 10 and left of the point with distance 5
    #only print 3 decimal places
    plt.annotate(str(round(euclidean_distance_list[last_iter], 3)), xy=(last_iter, euclidean_distance_list[last_iter]), xycoords='data',
                xytext=(-5, 10), textcoords='offset points',
                ha='center', va='bottom', fontsize=10, color='lightblue',
                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
                )
    #  

    # Show the plot
    plt.show()
    # #if save is true, save the plot
    # if save:
    #     plt.savefig('/home/lisa/bhartendu/kernal_convergence/power_method_plot.png', dpi=300, bbox_inches='tight')
def power_method_for_images(f, input_image, args_dict, max_iterations=1000, \
                            norm_tolerance=1e-8, dot_tolerance=1e-12, plot= True, verbose = True):
    
    m, n = input_image.shape
    #create 3 lists to store the gain in norm, euclidean distance, and dot product
    norm_gain_list = []
    euclidean_distance_list = []
    dot_product_list = []
    #initialize variables
    i = 0
    euclidean_distance = 1e+5
    cos_distance = -1
    #create a random image
    x = np.random.rand(m, n)
    #x to double
    x = x.astype(np.float64)
    x /= np.linalg.norm(x, 'fro')
    
    y = f( x, **args_dict)
    for i in range(max_iterations):
        y = f(x, **args_dict)
        #the norm of y is the gain in this step
        y_norm = np.linalg.norm(y, 'fro')
        x_new = y / y_norm 
        
        euclidean_distance = np.linalg.norm(x_new - x, 'fro')
        cos_distance = np.sum(np.multiply(x_new,x))

        #append the gain in norm to the list
        norm_gain_list.append(y_norm)
        #append the euclidean distance to the list
        euclidean_distance_list.append(euclidean_distance)
        #append the dot product to the list
        dot_product_list.append(cos_distance)
        # if euclidean_distance < norm_tolerance:
        #use math.isclose to get close to zero with norm_tolerance
        if math.isclose(euclidean_distance, 0, abs_tol=norm_tolerance):
            #print the return reason
            #that the norm of the difference between x_new and x is less than tolerance
            # print('Converged at iteration', i, 'because the norm of the difference between x_new and x is less than tolerance')
            #print converged as the norm of the difference between x_new and x is less than tolerance
            if verbose:
                print('Converged because the norm of the difference between x_new and x is less than tolerance')
            break
        #as both x_new and x are unit norm images, we can check if the cosine of the angle between them is
        #  less than tolerance, and the dot product of the two vectors is the cosine of the angle between them
        #just to stop we can check if the dot product is less than more than 1-tolerance
        #first take sum and then take sum, to get the dot product of two matrices
        # if np.sum(np.multiply(x_new,x)) >= 1-dot_tolerance:
        #we will test if the dot product close to 1, then we will stop
        #use the tolerance as dot_tolerance, and function for checking close
        
        if math.isclose(cos_distance, 1, abs_tol=dot_tolerance, rel_tol=dot_tolerance):
            #print the return reason
            #that the dot product is greater than 1-tolerance
            if verbose:
                print('Converged because the dot product is greater than 1-tolerance')
            break
        x = x_new
    #print iterations at which the power method is stopped
    #if i==max_iterations:, we print not converged

    if verbose:
        if i==max_iterations:
            print('Not converged')

        print('iteration', i)
        #print euclidean distance
        print('Euclidean distance', euclidean_distance)
        #also print the dot product of x_new and x
        print('cosine similarity', cos_distance)

    #PLOTTING
    if plot:
        #plot the dictionary
        plot_power_method(norm_gain_list, euclidean_distance_list, dot_product_list)
    # return np.linalg.norm(y, 'fro')
    return np.linalg.norm(y, 'fro'), x_new


def second_largest_eigenvalue(f, input_image, args_dict, max_iterations=1000, 
                              norm_tolerance=1e-8, dot_tolerance=1e-12, plot=False, verbose=False):
    
    # First, find the largest eigenvalue and its eigenvector
    largest_eigenvalue, largest_eigenvector = power_method_for_images(f= f, input_image=input_image, args_dict=args_dict, \
                                                                      max_iterations=max_iterations, norm_tolerance=norm_tolerance, \
                                                                      dot_tolerance=dot_tolerance, plot=plot, verbose=verbose)
    
    # Normalize the largest eigenvector
    largest_eigenvector /= np.linalg.norm(largest_eigenvector, 'fro')

    # Deflate the matrix
    def deflated_f(x, **args_dict):
        y = f(x, **args_dict)
        y -= (largest_eigenvalue * np.dot(largest_eigenvector.flatten(), x.flatten()) )* largest_eigenvector
        return y

    # Now find the largest eigenvalue of the deflated function, which will be the second largest of the original
    second_largest, second_eigenvector = power_method_for_images(f = deflated_f, input_image=input_image, args_dict=args_dict, \
                                                                 max_iterations=max_iterations, norm_tolerance=norm_tolerance, \
                                                                 dot_tolerance=dot_tolerance, plot=plot, verbose=verbose)   

    return second_largest, second_eigenvector


def smallest_eigenvalue(f, input_image, args_dict, max_iterations=1000, 
                        norm_tolerance=1e-8, dot_tolerance=1e-12, plot=False, verbose=False):
    
    
    # Define a new function that applies (A - I)
    def shifted_f(x, **kwargs):
        y = f(x, **kwargs)
        return y - x  # This is equivalent to (A - I)x
    
    # Find the largest eigenvalue of the shifted function
    shifted_largest_eigenvalue, smallest_eigenvector = power_method_for_images(f = shifted_f, input_image=input_image, args_dict=args_dict, \
                                                                      max_iterations=max_iterations, norm_tolerance=norm_tolerance, \
                                                                      dot_tolerance=dot_tolerance, plot=plot, verbose=verbose)
    
    # The smallest eigenvalue of A is the largest eigenvalue of (A - I) plus 1
    smallest_eigenvalue = -shifted_largest_eigenvalue + 1
    
    return smallest_eigenvalue, smallest_eigenvector

#we will write power method for non symmetric functionals
#thus we will accept two functionals, one for the functional and one for the adjoint
#we will also accept the argument dictionary for both the functionals
#update rule: singular_vec_right = functional_adjoint(singular_vec_left, **args_dict_functional_adjoint)/ ||functional_adjoint(singular_vec_left, **args_dict_functional_adjoint)||_2
#singular_vec_left = functional(singular_vec_right, **args_dict_functional)/ ||functional(singular_vec_right, **args_dict_functional)||_2
#at the end of the iteration, we will return the dot product of singular_vec_left and functional(singular_vec_right, **args_dict_functional)
def power_method_for_images_non_symmetric(functional, functional_adjoint, image_height, image_width, \
     args_dict_functional, args_dict_functional_adjoint, max_iterations=1000, \
                            norm_tolerance=1e-8, dot_tolerance=1e-12, plot= True, verbose = False):  
    m, n = image_height, image_width
    #create 3 lists to store the gain in norm, euclidean distance, and dot product for u
    #u is left singular vector of functional
    norm_gain_list_u = []
    euclidean_distance_list_u = []
    dot_product_list_u = []
    #create 3 lists to store the gain in norm, euclidean distance, and dot product for v
    #v is right singular vector of functional
    norm_gain_list_v = []
    euclidean_distance_list_v = []
    dot_product_list_v = []
    #initialize variables
    
    euclidean_distance_u = 1e+5
    cos_distance_u = -1
    euclidean_distance_v = 1e+5
    cos_distance_v = -1
    #create a random images
    u = (np.random.rand(m, n)).astype(np.float64)
    u /= np.linalg.norm(u, 'fro')
    v = (np.random.rand(m, n)).astype(np.float64)
    v /= np.linalg.norm(v, 'fro')
    for i in range(max_iterations):
        #we will find the right singular vector of functional
        v_new = functional_adjoint(u, **args_dict_functional_adjoint)
        #the norm of v_new is the gain in this step
        v_norm = np.linalg.norm(v_new, 'fro')
        if math.isclose(v_norm, 0, abs_tol=1e-12):
            v_new = v_new
            if verbose:
                print('right singular vector is zero vector')
        v_new = v_new / v_norm
        #we will find the left singular vector of functional
        u_new = functional(v_new, **args_dict_functional)
        #the norm of u_new is the gain in this step
        u_norm = np.linalg.norm(u_new, 'fro')
        if math.isclose(u_norm, 0, abs_tol=1e-12):
            u_new = u_new
            if verbose:
                print('left singular vector is zero vector')
        u_new = u_new / u_norm
        #we will find the euclidean distance between u_new and u
        euclidean_distance_u = np.linalg.norm(u_new - u, 'fro')
        #we will find the euclidean distance between v_new and v
        euclidean_distance_v = np.linalg.norm(v_new - v, 'fro')
        #we will find the dot product between u_new and u
        cos_distance_u = np.sum(np.multiply(u_new,u))
        #we will find the dot product between v_new and v
        cos_distance_v = np.sum(np.multiply(v_new,v))
        #append the gain in norm to the list
        norm_gain_list_u.append(u_norm)
        #append the euclidean distance to the list
        euclidean_distance_list_u.append(euclidean_distance_u)
        #append the dot product to the list
        dot_product_list_u.append(cos_distance_u)
        #append the gain in norm to the list
        norm_gain_list_v.append(v_norm)
        #append the euclidean distance to the list
        euclidean_distance_list_v.append(euclidean_distance_v)
        #append the dot product to the list
        dot_product_list_v.append(cos_distance_v)
        # if euclidean_distance < norm_tolerance:
        #use math.isclose to get close to zero with norm_tolerance
        if math.isclose(euclidean_distance_u, 0, abs_tol=norm_tolerance) and \
            math.isclose(euclidean_distance_v, 0, abs_tol=norm_tolerance):
            #print the return reason
            #that the norm of the difference between x_new and x is less than tolerance
            # print('Converged at iteration', i, 'because the norm of the difference between x_new and x is less than tolerance')
            #print converged as the norm of the difference between x_new and x is less than tolerance
            if verbose:
                print('Converged because the norm of the difference between both left and right singular vectors is less than tolerance')
            break
        #as both x_new and x are unit norm images, we can check if the cosine of the angle between them is
        #  less than tolerance, and the dot product of the two vectors is the cosine of the angle between them
        if math.isclose(cos_distance_u, 1, abs_tol=dot_tolerance, rel_tol=dot_tolerance) and \
            math.isclose(cos_distance_v, 1, abs_tol=dot_tolerance, rel_tol=dot_tolerance):
            #print the return reason
            #that the dot product is greater than 1-tolerance
            if verbose:
                print('Converged because the dot product of two successive left and right singular vectors is greater than 1-tolerance')
            break
        u = u_new
        v = v_new
        if math.isclose(u_norm, 0, abs_tol=1e-12):
            break
        if math.isclose(v_norm, 0, abs_tol=1e-12):
            break
    #print iterations at which the power method is stopped
    #if i==max_iterations:, we print not converged
    if verbose:
        if i == max_iterations-1:
            print('Not converged')
        print('iteration', i)
        #print euclidean distance
        print('Euclidean distance for left singular vector', euclidean_distance_u)
        print('Euclidean distance for right singular vector', euclidean_distance_v)
        #also print the dot product of x_new and x
        print('cosine similarity for left singular vector', cos_distance_u)
        print('cosine similarity for right singular vector', cos_distance_v)
    #PLOTTING
    if plot:
        #plot the dictionary
        print('plotting for left singular vector')
        plot_power_method(norm_gain_list_u, euclidean_distance_list_u, dot_product_list_u)
        print('plotting for right singular vector')
        plot_power_method(norm_gain_list_v, euclidean_distance_list_v, dot_product_list_v)

    #we will return the dot product of u and functional(v, **args_dict_functional)
    return np.sum(np.multiply(u , functional(v, **args_dict_functional))), u, v
    






######################################################################################
######################################################################################
#------------------              POWER METHOD ENDS         ----------------------------