from pystar.base import BaseModel


class GridSearch(BaseModel):
    def __init__(self, model, fixed_params, search_params):
        self._models = generate_models(model, fixed_params, search_params)
        self._scores = None
        self._best_model = None

    def fit(self, x_train, y_train):
        self._scores = []
        self._best_model = (0, None)
        for model in self._models:
            model.fit(x_train, y_train)
            score = model.score(x_train, y_train)
            if score > self._best_model[0]:
                self._best_model = (score, model)
            self._scores.append((model.score(), model))
        return self.best_model.weights

    def predict(self, x_test):
        return self.best_model.predict(x_test)

    @property
    def best_model(self):
        return self._best_model[1]

    @classmethod
    def create(cls, x_train, y_train, model, fixed_params, search_params):
        grid_search = cls(model, fixed_params, search_params)
        grid_search.fit(x_train, y_train)


def generate_models(model, fixed_params, search_params):
    # TODO: Decide if the search should be over cross product of all search_params

    for param, search_space in search_params.items():
        for value in search_space:
            yield model(**fixed_params, **{param: value})
