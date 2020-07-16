import numpy as np

import pystar.linear_models.utils as utils
from pystar.base import BaseModel, RegressorMixin


class LinearRegressor(BaseModel, RegressorMixin):

    def __init__(self, step_size=1e-5, num_iterations=100):
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.weights = None
        self.train_costs = None

    def fit(self, x_train, y_train):
        x, y = utils.pre_process_data(x_train, y_train)

        self.weights, self.train_costs = utils.gradient_descent(
            x,
            y,
            step_size=self.step_size,
            num_iterations=self.num_iterations,
            derivative_func=LinearRegressor.derivative
        )
        return self.weights

    def predict(self, x_test):
        x = utils.pre_process_data(x_test)
        return utils.calculate_output(x, self.weights)

    @classmethod
    def create(cls, x_train, y_train, step_size=1e-5, num_iterations=100):
        model = cls(step_size=step_size, num_iterations=num_iterations)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def derivative(x, y, weights):
        outputs = utils.calculate_output(x, weights)
        errors = y - outputs
        return - 2 * np.dot(errors, x)

# model = LinearRegressor()
# model.fit(x, y)
# a = model.weights
# print(a) 1, 3
# model.fit(x1, y1)
# b = model.weights
# print(b) 2, 7
# print(a) 2, 7
