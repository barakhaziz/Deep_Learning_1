from typing import Dict, Tuple, List, Any
from sklearn.model_selection import train_test_split
import os
import timeit
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KERAS_BACKEND"] = "torch"
from keras_core.datasets import mnist


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
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
        # parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.1
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
    Performs forward propagation for a single layer with a linear transformation
    followed by an activation function (ReLU or Softmax).

    Args:
        A_prev (ndarray): Activations from the previous layer (or input data),
                          shape (size of previous layer, number of examples).
        W (ndarray): Weight matrix of shape (size of current layer, size of previous layer).
        b (ndarray): Bias vector of shape (size of current layer, 1).
        activation (str): Activation function to apply; either "relu" or "softmax".

    Returns:
        A (ndarray): Post-activation output of the current layer.
        cache (tuple): Tuple containing linear and activation caches,
                       used later for backpropagation.
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
        # w is 2-dim matrix
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]


        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
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
    Computes the cross-entropy loss for a multi-class classification problem,
    with optional L2 regularization.

    Args:
        AL (ndarray): Predicted probabilities, shape (num_classes, number of examples).
        Y (ndarray): Ground truth labels in one-hot encoded form, same shape as AL.
        parameters (dict): Dictionary containing model parameters (weights and biases).
        l2_regularization (bool): Whether to include L2 regularization in the loss computation.

    Returns:
        cost (float): The total loss (cross-entropy with optional L2 regularization).
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
    Applies batch normalization to the activation values of a layer.

    Args:
        A (ndarray): Activation values of shape (num_features, batch_size).

    Returns:
        NA (ndarray): Normalized activations computed using the batch mean and variance.
                      Formula: (A - mean) / sqrt(variance + epsilon)
    """
    # Calculate mean and variance across the batch dimension (axis=1)
    # A has shape (features, batch_size)
    mean = np.mean(A, axis=1, keepdims=True)  # Normalize across batch, for each feature
    var = np.var(A, axis=1, keepdims=True)  # Normalize across batch, for each feature

    # Apply normalization
    NA = (A - mean) / np.sqrt(var + EPSILON)

    return NA



def linear_backward(dZ: np.ndarray, cache: Tuple, l2_regularization: bool = False) -> Tuple:
    """
    Computes the gradients for the linear portion of backward propagation for a single layer.

    Args:
        dZ (ndarray): Gradient of the loss with respect to the linear output (Z) of the current layer.
        cache (tuple): Tuple containing (A_prev, W, b) from the forward pass of the current layer.
        l2_regularization (bool): If True, includes L2 regularization in the gradient calculation for W.

    Returns:
        dA_prev (ndarray): Gradient of the loss with respect to the activations from the previous layer.
        dW (ndarray): Gradient of the loss with respect to the weights of the current layer.
        db (ndarray): Gradient of the loss with respect to the bias of the current layer.
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
    Performs the backward pass for a single layer (LINEAR -> ACTIVATION).

    This function first computes dZ using the derivative of the specified activation function,
    then calculates the gradients using the linear_backward function.

    Args:
        dA (ndarray): Gradient of the loss with respect to the post-activation output of the current layer.
        cache (tuple): Tuple containing (linear_cache, activation_cache) from the forward pass.
        activation (str): Activation function used in the forward pass ("relu" or "softmax").
        l2_regularization (bool): Whether to include L2 regularization in the weight gradients.

    Returns:
        dA_prev (ndarray): Gradient of the loss with respect to the activation from the previous layer.
        dW (ndarray): Gradient of the loss with respect to the current layer's weights.
        db (ndarray): Gradient of the loss with respect to the current layer's biases.
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
    Computes the gradient of the ReLU activation during backpropagation.

    Args:
        dA (ndarray): Gradient of the loss with respect to the activation output.
        activation_cache (dict): Contains 'Z', the pre-activation input from the forward pass.

    Returns:
        dZ (ndarray): Gradient of the loss with respect to Z.
    """
    Z = activation_cache["Z"]
    dZ = np.where(Z > 0, dA, np.zeros_like(dA))
    return dZ



def softmax_backward(dA: np.ndarray, activation_cache: Dict) -> np.ndarray:
    """
    Computes the gradient of the softmax activation for the output layer during backpropagation.

    Args:
        dA (ndarray): Gradient of the loss with respect to the softmax output (typically equals AL).
        activation_cache (dict): Contains 'Y' (true labels) from the forward pass.

    Returns:
        dZ (ndarray): Gradient of the loss with respect to Z (pre-activation).
                      For softmax with cross-entropy, dZ = AL - Y.
    """
    dZ = dA - activation_cache["Y"]
    return dZ


def L_model_backward(AL: np.ndarray, Y: np.ndarray, caches: List, l2_regularization: bool = False) -> Dict[str, Any]:
    """
    Implements the full backward propagation for an L-layer neural network.

    The final layer uses softmax, while all preceding layers use ReLU.
    Gradients are computed for each layer in reverse order and stored in a dictionary.

    Args:
        AL (ndarray): Output probabilities from the forward pass (shape: num_classes × num_examples).
        Y (ndarray): True labels (one-hot encoded), same shape as AL.
        caches (list): List of cache tuples (linear_cache, activation_cache) for each layer.
        l2_regularization (bool): If True, includes L2 regularization in the weight gradients.

    Returns:
        grads (dict): Dictionary containing gradients for all layers.
                      Keys follow the format:
                      - grads["dA{l}"], grads["dw{l}"], grads["db{l}"]
    """
    grads = dict()
    n_layers = len(caches)

    for l in reversed(range(n_layers)):
        current_cache = caches[l]
        linear_cache, activation_cache = current_cache

        # Create a new dictionary with Z from activation_cache
        activation_cache_dict = {"Z": activation_cache}

        if l == n_layers - 1:
            # For the output layer, add Y to the activation cache dictionary
            activation_cache_dict.update({"Y": Y})
            dA_prev, dW, db = linear_activation_backward(AL, (linear_cache, activation_cache_dict), "softmax",
                                                         l2_regularization)
        else:
            dA = dA_prev
            dA_prev, dW, db = linear_activation_backward(dA, (linear_cache, activation_cache_dict), "relu",
                                                         l2_regularization)

        grads["dA" + str(l)] = dA_prev
        grads["dw" + str(l)] = dW
        grads["db" + str(l)] = db

    return grads

# 2.f
def update_parameters(parameters: Dict, grads: Dict, learning_rate: float) -> Dict:
    """
    Updates the model parameters using gradient descent.

    Args:
        parameters (dict): Dictionary containing the network’s current weights and biases.
                           Keys follow the format "W1", "b1", ..., "WL", "bL".
        grads (dict): Dictionary containing gradients computed from backpropagation.
                      Keys follow the format "dw0", "db0", ..., "dw{L-1}", "db{L-1}".
        learning_rate (float): Learning rate for the gradient descent update.

    Returns:
        parameters (dict): Dictionary with updated weights and biases.
    """
    # Count number of layers (based on weights)
    L = len([key for key in parameters if key.startswith('W')])

    # Update each layer's parameters
    for l in range(1, L + 1):
        w_key = 'W' + str(l)
        b_key = 'b' + str(l)
        dw_key = 'dw' + str(l - 1)  # Adjust for zero-indexing in gradients
        db_key = 'db' + str(l - 1)  # Adjust for zero-indexing in gradients

        parameters[w_key] = parameters[w_key] - learning_rate * grads[dw_key]
        parameters[b_key] = parameters[b_key] - learning_rate * grads[db_key]

    return parameters


# 3.a
def L_layer_model(
        X: np.ndarray,
        Y: np.array,
        layer_dims: List,
        learning_rate: float,
        num_iterations: int,
        batch_size: int = 128,
        use_batchnorm: bool = False,
        l2_regularization: bool = False
):
    """
    Trains an L-layer fully connected neural network using forward and backward propagation,
    with optional batch normalization and L2 regularization.

    The architecture consists of (L-1) layers with ReLU activation followed by a final softmax layer.
    Training is performed using mini-batch gradient descent, with evaluation on a validation split.

    The function executes the training loop with the following steps:
        initialize -> forward pass -> compute cost -> backward pass -> update parameters

    Args:
        X (ndarray): Input data of shape (input_size, number_of_examples).
        Y (ndarray): One-hot encoded true labels of shape (num_classes, number_of_examples).
        layer_dims (List[int]): List defining the number of units in each layer, including input and output.
        learning_rate (float): Learning rate for gradient descent.
        num_iterations (int): Total number of parameter updates (not epochs).
        batch_size (int): Number of examples per training batch (default: 128).
        use_batchnorm (bool): If True, applies batch normalization after ReLU activations.
        l2_regularization (bool): If True, includes L2 regularization in cost and gradient calculations.

    Returns:
        parameters (dict): Learned network parameters (weights and biases for each layer).
        costs (list): Cross-entropy cost values recorded every 100 iterations.
        accuracy_histories (dict): Dictionary with training and validation accuracy recorded every 100 iterations.
                                   Keys: "train", "validation".
    """
    parameters = initialize_parameters(layer_dims)
    costs = []
    accuracy_histories = {"train": [], "validation": []}

    # Get validation set that is 20% of the training set - the split is stratified by the labels
    X_train, X_val, Y_train, Y_val = train_test_split(
        X.T, Y.T, test_size=0.2, random_state=42, stratify=Y.T
    )
    X_train, X_val, Y_train, Y_val = X_train.T, X_val.T, Y_train.T, Y_val.T

    # Training loop
    num_examples = X_train.shape[1]
    i = 0  # iteration counter
    patience, patience_counter = 20, 0
    epoch_counter = 1
    best_validation_accuracy = 0
    while i < num_iterations:
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            X_batch, Y_batch = X_train[:, batch_start: batch_end], Y_train[:, batch_start: batch_end]

            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm=use_batchnorm)
            grads = L_model_backward(AL, Y_batch, caches, l2_regularization)
            parameters = update_parameters(parameters, grads, learning_rate)

            if i % 100 == 0:
                cost = compute_cost(AL, Y_batch, parameters, l2_regularization)
                costs.append(cost)
                train_accuracy = Predict(X_train, Y_train, parameters, use_batchnorm)
                val_accuracy = Predict(X_val, Y_val, parameters, use_batchnorm)

                if val_accuracy > best_validation_accuracy:
                    best_validation_accuracy = val_accuracy
                    patience_counter = 0

                accuracy_histories["train"].append(train_accuracy)
                accuracy_histories["validation"].append(val_accuracy)

                print(
                    f"Epoch_{epoch_counter}/Iteration_{i}: training_cost = {round(cost, 3)}"
                    f" | training_accuracy = {round(train_accuracy * 100, 3)}%"
                    f" | validation_accuracy = {round(val_accuracy * 100, 3)}%"
                    f" | best validation_accuracy = {round(best_validation_accuracy * 100, 3)}%"
                )

                # Check EarlyStopping - if best validation accuracy >> current val accuracy then score isn't improving
                if i > 0 and best_validation_accuracy - val_accuracy > EPSILON and patience_counter > patience:
                    print("Early stopping reached!")
                    return parameters, costs, accuracy_histories
                else:
                    patience_counter += 1

                if i == num_iterations:
                    return parameters, costs, accuracy_histories

            i += 1

        # The inner loop represents a full epoch and when it is complete the epoch counter is increased
        epoch_counter += 1

    return parameters, costs, accuracy_histories


# 3.b
def Predict(X: np.ndarray, Y: np.array, parameters: Dict, use_batchnorm: bool) -> float:
    """
    Computes the classification accuracy of the trained neural network on the given dataset.

    Args:
        X (ndarray): Input data of shape (input_size, number_of_examples).
        Y (ndarray): Ground truth labels in one-hot encoded format, shape (num_classes, number_of_examples).
        parameters (dict): Trained model parameters (weights and biases).
        use_batchnorm (bool): Whether to apply batch normalization during the forward pass.

    Returns:
        accuracy (float): Proportion of correctly classified examples (value between 0 and 1).
    """
    num_examples = X.shape[1]
    AL, _ = L_model_forward(X, parameters, use_batchnorm=use_batchnorm)
    labels, y_pred = np.argmax(Y, axis=0), np.argmax(AL, axis=0)
    accuracy = np.sum(y_pred == labels) / num_examples
    return accuracy


def get_mnist():
    """
    Loads and preprocesses the MNIST dataset for training and testing.

    Processing steps:
      - Normalizes pixel values to the [0, 1] range.
      - Flattens each 28x28 image into a column vector.
      - Applies one-hot encoding to the labels.

    Returns:
        x_train (ndarray): Flattened and normalized training images, shape (784, num_train_samples).
        x_test (ndarray): Flattened and normalized test images, shape (784, num_test_samples).
        y_train (ndarray): One-hot encoded training labels, shape (10, num_train_samples).
        y_test (ndarray): One-hot encoded test labels, shape (10, num_test_samples).
    """
    # Load dataset
    (x_train_val, y_train_val), (x_test, y_test) = mnist.load_data()
    # normalize dataset by 255 (max value)
    x_train_val = x_train_val / 255.0
    x_test = x_test / 255.0

    # Flatten the input of train_val and test sets
    img_h, img_w = x_train_val.shape[1], x_train_val.shape[2]
    x_train = x_train_val.reshape((x_train_val.shape[0], img_h * img_w), order='F').T
    x_test = x_test.reshape((x_test.shape[0], img_h * img_w), order='F').T

    # Apply onehot encoding to y
    onehot = OneHotEncoder(sparse_output=False)
    y_train = onehot.fit_transform(y_train_val.reshape(-1, 1)).T
    y_test = onehot.transform(y_test.reshape(-1, 1)).T

    return x_train, x_test, y_train, y_test


def main():
    # Set consistent parameters
    seed = 42
    batch_size = 64
    learning_rate = 0.009
    num_iterations = 100000
    np.random.seed(seed)

    # Load and preprocess MNIST data
    x_train, x_test, y_train, y_test = get_mnist()
    input_dim = x_train.shape[0]
    layer_dims = [input_dim, 20, 7, 5, 10]

    # Define experiment settings
    experiment_settings = [
        {"name": "No BN, No L2", "use_batchnorm": False, "l2_regularization": False},
        {"name": "Yes BN, No L2", "use_batchnorm": True, "l2_regularization": False},
        {"name": "No BN, Yes L2", "use_batchnorm": False, "l2_regularization": True},
    ]

    # Store results for comparison
    all_params, all_costs, all_acc_histories = [], [], []

    # Run all experiments
    for exp in experiment_settings:
        print(f"\nRunning Experiment: {exp['name']}")
        start_time = timeit.default_timer()

        parameters, costs, acc_histories = L_layer_model(
            X=x_train,
            Y=y_train,
            layer_dims=layer_dims,
            learning_rate=learning_rate,
            num_iterations=num_iterations,
            batch_size=batch_size,
            use_batchnorm=exp["use_batchnorm"],
            l2_regularization=exp["l2_regularization"]
        )

        end_time = timeit.default_timer()
        print(f"Training duration: {round(end_time - start_time, 3)} seconds")
        print(
            f"Testing Accuracy: {Predict(X=x_test, Y=y_test, parameters=parameters, use_batchnorm=exp['use_batchnorm'])}")

        all_params.append(parameters)
        all_costs.append(costs)
        all_acc_histories.append(acc_histories)

    # Create and display visualizations
    create_visualizations(all_params, all_costs, all_acc_histories, experiment_settings, layer_dims)


def create_visualizations(all_params, all_costs, all_acc_histories, experiment_settings, layer_dims):
    # Colors for consistency
    colors = ['royalblue', 'forestgreen', 'firebrick', 'darkorange']

    # 1. Combined Cost Plot
    plt.figure(figsize=(10, 6))
    for i, costs in enumerate(all_costs):
        plt.plot(np.squeeze(costs), label=experiment_settings[i]["name"], color=colors[i])
    plt.ylabel('Cost')
    plt.xlabel('Number of iterations (·10²)')
    plt.title("Cost vs. Iterations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Combined Accuracy Plot
    plt.figure(figsize=(12, 8))
    for i, acc_history in enumerate(all_acc_histories):
        iteration_steps = list(range(0, len(acc_history['train']) * 100, 100))
        plt.plot(iteration_steps, acc_history['train'], label=f"{experiment_settings[i]['name']} - Train",
                 color=colors[i], linestyle='-')
        plt.plot(iteration_steps, acc_history['validation'], label=f"{experiment_settings[i]['name']} - Val",
                 color=colors[i], linestyle='--')
    plt.title('Accuracy Comparison')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # 3. Weight Distribution Comparison
    n_layers = len(layer_dims) - 1
    fig, axes = plt.subplots(len(all_params), n_layers, figsize=(20, 16))
    fig.suptitle('Weight Distributions by Layer and Experiment', fontsize=16)

    for exp_idx, parameters in enumerate(all_params):
        for layer_idx in range(n_layers):
            layer_num = layer_idx + 1
            weights = parameters['W' + str(layer_num)].flatten()
            mse = np.mean(np.square(weights))

            ax = axes[exp_idx, layer_idx]
            ax.hist(weights, bins=30, edgecolor='black', color=colors[exp_idx], alpha=0.7)
            ax.set_title(f"{experiment_settings[exp_idx]['name']} - Layer {layer_idx + 1}")
            ax.text(0.05, 0.90, f'MSE: {mse:.6f}', transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()