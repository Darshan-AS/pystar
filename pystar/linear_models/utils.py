import numpy as np


def pre_process_data(x, y=None):
    x, y = convert_to_numpy(x), convert_to_numpy(y)
    x = np.column_stack((np.ones(x.shape[0]), x))
    return (x, y) if isinstance(y, np.ndarray) else x


def convert_to_numpy(data):
    return data if data is None or isinstance(data, np.ndarray) else data.to_numpy()


def calculate_output(x, weights):
    return np.dot(x, weights)


def gradient_descent(x_train, y_train, step_size, num_iterations, derivative_func):
    weights = np.zeros(x_train.shape[1])
    rss_train_costs = np.zeros(num_iterations)

    for i in range(num_iterations):
        rss_train_costs[i] = rss(y_train, calculate_output(x_train, weights))
        weights = weights - step_size * derivative_func(x_train, y_train, weights) / num_iterations
    rss_train_costs[-1] = rss(y_train, calculate_output(x_train, weights))

    return weights, rss_train_costs


def coordinate_descent(x_train, y_train, num_iterations, derivative_func):
    weights = np.zeros(x_train.shape[1])
    rss_train_costs = np.zeros(num_iterations)

    for i in range(num_iterations):
        for j in range(1, weights.shape[0]):
            pass
    # TODO: Implement coordinate descent
