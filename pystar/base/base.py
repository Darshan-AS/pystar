from abc import ABC, abstractmethod

from pystar.metrics import r2_score, rss


class BaseModel(ABC):

    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass


class RegressorMixin:
    def r2_score(self: BaseModel, x, y):
        y_pred = self.predict(x)
        return r2_score(y, y_pred)

    def rss_score(self: BaseModel, x, y):
        y_pred = self.predict(x)
        return rss(y, y_pred)

    score = rss_score
