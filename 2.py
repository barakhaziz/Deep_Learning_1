import numpy as np
from typing import Tuple, List, Dict, Any

np.random.seed(2)
EPSILON = 1e-6


def linear_backward(dZ: np.ndarray, cache: Tuple, l2_regularization: bool = False) -> Tuple:
    """
    Computes the linear part of the backward propagation process of a single layer
    :param dZ: The gradient of the cost with respect to the linear output of the current layer
    :param cache: Tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :param l2_regularization: boolean - indicates if l2 regularization should be used
    :return: dA_prev - Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :return: dW - Gradient of the cost with respect to W (current layer l), same shape as W
    :return: db - Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T) + ((EPSILON / m) * W if l2_regularization else 0)
    # not sure about dividing with m
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def linear_activation_backward(dA: np.ndarray, cache: Tuple, activation: str, l2_regularization: bool = False) -> Tuple:
    """
    Compute the backward propagation for the LINEAR->ACTIVATION layer.
    The function first computes dZ and then applies the linear_backward function
    :param dA: post activation gradient of the current layer
    :param cache:  cache contains both the linear and the activations cache
    :param activation: The activation function to be used (a string, either “softmax” or “relu”)
    :param l2_regularization: boolean - indicates if l2 regularization should be used
    :return: dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    :return: dW – Gradient of the cost with respect to W (current layer l), same shape as W
    :return: db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == 'relu':
        back_activation = relu_backward
    elif activation == 'softmax':
        back_activation = softmax_backward
    else:
        raise ValueError(f"activation {activation} isn't implemented")
    dZ = back_activation(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache, l2_regularization)
    return dA_prev, dW, db


def relu_backward(dA: np.ndarray, activation_cache: Dict) -> np.ndarray:
    """
    Compute backward propagation for a ReLU unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache["Z"]
    dZ = np.where(Z > 0, dA, np.zeros_like(dA))
    return dZ



def softmax_backward(dA: np.ndarray, activation_cache: Dict) -> np.ndarray:
    """
    Compute backward propagation for a ReLU unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    dZ = dA - activation_cache["Y"]
    return dZ
