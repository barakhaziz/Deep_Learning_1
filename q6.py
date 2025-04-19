import os
import timeit
import numpy as np
from q4 import get_mnist
from q3 import L_layer_model, Predict
import matplotlib.pyplot as plt
import pandas as pd

os.environ["KERAS_BACKEND"] = "torch"

np.random.seed(0)
def main():
    x_train, x_test, y_train, y_test = get_mnist()
    input_dim = x_train.shape[0]

    # Q4.b Batch Normalization = True
    layer_dims = [input_dim, 20, 7, 5, 10]  # 4 layers (aside from the input layer), with the following sizes: 20,7,5,10

    start_time_train = timeit.default_timer()
    parameters, costs, accuracy_histories = L_layer_model(
                                                            X=x_train,
                                                            Y=y_train,
                                                            layer_dims=layer_dims,
                                                            learning_rate=0.009,  # Use a learning rate of 0.009
                                                            num_iterations=1000000,
                                                            batch_size=64,
                                                            use_batchnorm=False,
                                                            l2_regularization=True
                                                        )
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time_train
    print('Training duration: ', str(round(elapsed_time, 3)))

    start_time_test = timeit.default_timer()
    print('Testing Accuracy: ',
          Predict(
            X=x_test,
            Y=y_test,
            parameters=parameters,
            use_batchnorm=False))
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time_test

    print('Test duration: ', str(round(elapsed_time, 3)), '\nTotal duration: ', str(round(end_time - start_time_train, 3)))

    train_acc_history = accuracy_histories['train']
    val_acc_history = accuracy_histories['validation']

    start_time_test = timeit.default_timer()
    print('Testing Accuracy: ',
          Predict(
              X=x_test,
              Y=y_test,
              parameters=parameters,
              use_batchnorm=False))
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time_test

    print('Test duration: ', str(round(elapsed_time, 3)), '\nTotal duration: ',
          str(round(end_time - start_time_train, 3)))

    n_layers = len(layer_dims) - 1
    fig, axes = plt.subplots(1, n_layers, figsize=(20, 5))
    fig.suptitle('Distribution of Weight Size by Layer - With L2 Regularization and Without Batch Normalization')

    weights_means = []
    for i in range(n_layers):
        layer_num = i + 1  # Parameters are indexed from 1
        weights = parameters['W' + str(layer_num)].flatten()
        weights_means.append(np.mean(np.square(weights)))
        ax = axes[i]  # plot weight by layer
        ax.hist(weights, bins=30, edgecolor='black')
        if i == 0:
            x = (-0.7, 0.7)
            y = (0, 2100)
        elif i == 1:
            x = (-1.5, 1.5)
            y = (0, 21)
        elif i == 2:
            x = (-1.5, 3)
            y = (0, 7.5)
        else:
            x = (-2.5, 1.75)
            y = (0, 8)
        ax.set_xlim(x),
        ax.set_ylim(y),
        ax.set_title(f'Layer {i + 1} Weights')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print(pd.DataFrame({'layer': np.arange(n_layers) + 1, 'mse': weights_means}))

    # Accuracy over iterations plot
    plt.figure(figsize=(8, 5))
    iteration_steps = list(range(0, len(train_acc_history) * 100, 100))
    plt.plot(iteration_steps, train_acc_history, label='Training Accuracy', color='royalblue')
    plt.plot(iteration_steps, val_acc_history, label='Validation Accuracy', color='darkorange')
    plt.title('Q6 - Yes L2 Regularization and No Batch Normalization')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    # Annotate final values with slight vertical offset
    plt.text(iteration_steps[-1], train_acc_history[-1] + 0.005, f"{train_acc_history[-1] * 100:.2f}%", va='bottom')
    plt.text(iteration_steps[-1], val_acc_history[-1] - 0.005, f"{val_acc_history[-1] * 100:.2f}%", va='top')
    plt.tight_layout()

    # Cost over iterations plot
    plt.figure(figsize=(6, 4))
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Number of iterations (·10²)')
    plt.title("Cost vs. Iterations")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()