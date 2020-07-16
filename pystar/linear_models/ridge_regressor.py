from functools import partial

import numpy as np

import pystar.linear_models._utils as utils
from pystar.base.base import BaseModel, RegressorMixin


class RidgeRegressor(BaseModel, RegressorMixin):

    def __init__(self, step_size=1e-10, num_iterations=100, l2_penalty=1e5):
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.l2_penalty = l2_penalty
        self.weights = None
        self.train_costs = None

    def fit(self, x_train, y_train):
        x, y = utils.pre_process_data(x_train, y_train)

        self.weights, self.train_costs = utils.gradient_descent(
            x,
            y,
            step_size=self.step_size,
            num_iterations=self.num_iterations,
            derivative_func=partial(RidgeRegressor.derivative, l2_penalty=self.l2_penalty)
        )
        return self.weights

    def predict(self, x_test):
        x = utils.pre_process_data(x_test)
        return utils.calculate_output(x, self.weights)

    @classmethod
    def create(cls, x_train, y_train, step_size=1e-10, num_iterations=100, l2_penalty=1e5):
        model = cls(step_size=step_size, num_iterations=num_iterations, l2_penalty=l2_penalty)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def derivative(x, y, weights, l2_penalty):
        outputs = utils.calculate_output(x, weights)
        errors = y - outputs
        return - 2 * np.dot(errors, x) + 2 * l2_penalty * weights
