def k_fold(x, y=None, k=2, **kwargs):
    assert x.shape[0] == y.shape[0]

    for i in range(0, x.shape[0], k):
        x_validate, y_validate = x[i: i + k], y[i: i + k]
        x_train, y_train = x[:i] + x[k:], y[:i] + y[k:]
        yield x_train, y_train, x_validate, y_validate


def leave_one_out(x, y=None, **kwargs):
    yield from k_fold(x, y, k=1, **kwargs)
