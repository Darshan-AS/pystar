from pystar.model_selection._splitters import k_fold


def cross_validate(model, x, y, k, splitter=k_fold, splitter_args=None):
    score = 0
    for x_train, y_train, x_validate, y_validate in splitter(x, y, k=k, **splitter_args):
        m = model()
        m.fit(x_train, y_train)
        score += m.score(x_validate, y_validate)
    return score / k
