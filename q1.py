import numpy as np
from typing import Dict, Tuple, List

np.random.seed(2)
EPSILON = 1e-6

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
       # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
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
    Implements the softmax activation function for multi-class classification.
    
    Arguments:
    Z -- the linear component of the activation function
    
    Returns:
    A -- output probabilities, same shape as Z
    cache -- contains Z for use in backpropagation
    """
    
    # Subtract max for numerical stability
    Z_stable = Z - np.max(Z, axis=0, keepdims=True)
    
    # Calculate softmax: exp(z_i) / sum(exp(z_j))
    exp_values = np.exp(Z_stable)
    A = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    
    # Save Z for backpropagation
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implements the ReLU activation function.
    
    Arguments:
    Z -- the linear component of the activation function
    
    Returns:
    A -- output activations, same shape as Z
    activation_cache -- contains Z for use in backpropagation
    """
    
    # ReLU function: max(0,Z)
    A = np.maximum(0, Z)
    
    # Store Z for backpropagation
    activation_cache = Z
    
    return A, activation_cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: (size of current layer, size of previous layer)
    b -- bias vector: (size of current layer, 1)
    activation -- the activation function to be used (string): "softmax" or "relu"
    
    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a joint dictionary containing both linear_cache and activation_cache;
             stored for computing the backward pass efficiently
    """
    
    # First, perform the linear computation
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    # Then apply the activation function
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError(f"Activation function {activation} not supported. Use 'softmax' or 'relu'.")
    
    # Create a joint cache that combines both caches
    cache = (linear_cache, activation_cache)
    
    return A, cache

def L_model_forward(X, parameters, use_batchnorm):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- dictionary containing the parameters W and b for each layer
    use_batchnorm -- boolean flag used to determine whether to apply batchnorm after activation
                     (Note: this option needs to be set to "False" in Section 3 and "True" in Section 4)
    
    Returns:
    AL -- last post-activation value (output of the softmax)
    caches -- list of all the cache objects generated by the linear_activation_forward function
    """
    
    caches = []
    A = X
    L = len(parameters) // 2  # Number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1)
    # The first L-1 layers use ReLU activation
    for l in range(1, L):
        A_prev = A
        
        # Get parameters for current layer
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        # Forward propagation for current layer
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        
        # Add cache to list
        caches.append(cache)
        
        # Apply batch normalization if specified
        # Note: Batch normalization implementation is not included here
        # It would be applied after activation
        if use_batchnorm:
            A = apply_batchnorm(A)
    
    # Implement LINEAR -> SOFTMAX for the last layer (Lth layer)
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    
    # Forward propagation for the output layer
    AL, cache = linear_activation_forward(A, W, b, "softmax")
    
    # Add cache to list
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL: np.ndarray, Y: np.ndarray, parameters: Dict, l2_regularization: bool = False) -> float:
    """
    Compute cost function categorical cross-entropy loss
    :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :param parameters: the W and b parameters of each layer
    :param l2_regularization: boolean - indicates if l2 regularization should be used
    :return: cost – the cross-entropy cost
    """
    n_examples = Y.shape[1]
    cost = -1 / n_examples * sum(np.sum(np.multiply(Y, np.log(AL + EPSILON)), axis=0))

    # l2 Norm
    if l2_regularization:
        l2_cost = sum([np.sum(np.square(layer[0])) for layer in parameters.values()])
        l2_cost = (EPSILON / (2 * n_examples)) * l2_cost
        cost += l2_cost
    return cost


def apply_batchnorm(A: np.ndarray) -> np.ndarray:
    """
    Performs batchnorm on the received activation values of a given layer
    :param A: The activation values of a given layer
    :return: NA - the normalized activation values, based on the formula learned in class

    """
    # Calculate mean and variance across the batch dimension (axis=1)
    # A has shape (features, batch_size)
    mean = np.mean(A, axis=1, keepdims=True)  # Normalize across batch, for each feature
    var = np.var(A, axis=1, keepdims=True)    # Normalize across batch, for each feature
    
    # Apply normalization
    NA = (A - mean) / np.sqrt(var + EPSILON)
    
    return NA