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