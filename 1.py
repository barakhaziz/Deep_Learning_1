import numpy as np

def initialize_parameters(layer_dims):
    """
    Initialize neural network parameters (weights W and biases b) for all layers.
    
    Arguments:
    layer_dims -- array/list containing the dimensions of each layer in the network
                 layer_dims[0] = input layer size (n[0])
                 layer_dims[L] = output layer size (n[L]) where L is the total number of layers
    
    Returns:
    parameters -- dictionary containing the initialized parameters:
                 parameters['W' + str(l)] = weight matrix of shape (layer_dims[l], layer_dims[l-1])
                 parameters['b' + str(l)] = bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims) - 1  # number of layers (excluding input layer)
    
    for l in range(1, L + 1):
        # Initialize weights using random normal distribution
        # He initialization scaling factor for better training with deep networks
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        
        # Initialize biases with zeros
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters
