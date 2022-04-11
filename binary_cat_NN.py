import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from binary_cat_LR import load_dataset

# matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# GRADED FUNCTION: n_layer_model

# def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False): #lr was 0.009
#     """
#     Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
#     Arguments:
#     X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
#     Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
#     layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
#     learning_rate -- learning rate of the gradient descent update rule
#     num_iterations -- number of iterations of the optimization loop
#     print_cost -- if True, it prints the cost every 100 steps
    
#     Returns:
#     parameters -- parameters learnt by the model. They can then be used to predict.
#     """

#     np.random.seed(1)
#     costs = []                         # keep track of cost
    
#     # Parameters initialization.
#     ### START CODE HERE ###
#     parameters = initialize_parameters_deep(layers_dims)
#     ### END CODE HERE ###
    
#     # Loop (gradient descent)
#     for i in range(0, num_iterations):

#         # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
#         AL, caches = L_model_forward(X, parameters)
#         # Compute cost.
#         cost = compute_cost(AL, Y)
#         # Backward propagation.
#         grads = L_model_backward(AL, Y, caches)
#         # Update parameters.
#         parameters = update_parameters(parameters, grads, learning_rate)
                
#         # Print the cost every 100 training example
#         if print_cost and i % 100 == 0:
#             print ("Cost after iteration %i: %f" % (i, cost))
#         if print_cost and i % 100 == 0:
#             costs.append(cost)
            
#     # plot the cost
#     plt.plot(np.squeeze(costs))
#     plt.ylabel('cost')
#     plt.xlabel('iterations (per tens)')
#     plt.title("Learning rate =" + str(learning_rate))
#     plt.show()
    
#     return parameters

def main():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
    # print(train_set_x_orig.shape)
    m_train = train_x_orig.shape[0]     # numbers of training exemples
    num_px = train_x_orig.shape[1]      # (num_px * num_px) is the size of each figure
    m_test = test_x_orig.shape[0]       # numbers of test exemples

    # Reshape the training and test examples, flatten them into vectors
    # The faltten vectors dimention should be (size of figure, number of exemples) 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    print(train_x_flatten.shape)
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255