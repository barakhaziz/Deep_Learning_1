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

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: (size of current layer, size of previous layer)
    b -- bias vector: (size of current layer, 1)
    
    Returns:
    Z -- the linear component of the activation function (i.e., the value before applying the non-linear function)
    cache -- a dictionary containing "A", "W", "b" for efficient computation of the backward pass
    """
    
    # Compute the linear transformation: Z = W·A + b
    Z = np.dot(W, A) + b
    
    # Create a cache containing A, W, b for use in the backward pass
    cache = (A, W, b)
    
    return Z, cache

def softmax(Z):
    """
    Implements the softmax activation function.
    
    Arguments:
    Z -- the linear component of the activation function, numpy array of any shape
    
    Returns:
    A -- output of softmax, same shape as Z
    cache -- a dictionary containing "Z" for efficient computation of the backward pass
    """
    
    # Shift Z for numerical stability (prevents overflow)
    # Subtracting the maximum value from each example (column)
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    
    # Compute the exponential of all values
    exp_Z = np.exp(Z_shifted)
    
    # Compute the softmax by normalizing each column (each example)
    # Sum across rows (classes) for each example
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    # Store Z in the cache for backpropagation
    cache = Z
    
    return A, cache
