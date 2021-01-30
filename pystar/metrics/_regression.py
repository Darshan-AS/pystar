import numpy as np


def r2_score(y_true, y_pred):
    raise NotImplementedError()


def rss(y_true, y_pred):
    errors = y_true - y_pred
    return np.dot(errors, errors)
