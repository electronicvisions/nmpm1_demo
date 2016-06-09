import contextlib

import matplotlib.pyplot as plt


@contextlib.contextmanager
def figure(**kwargs):
    """
    Create a figure with a set of subplots already made.

    If the `figsize` argument is not specified, the dimensions will be
    calculated based on optional parameters `figwidth`, `figheight`
    and `figratio` (if given).
    If neither `nrows` or `ncols` have been specified, a single axis
    will be returned; else the returned axes will always be indexable.

    ---------- This function uses matplotlib.pyplot.figure, ----------
                 whose documentation is reproduced below.
    """
    height = kwargs.pop('figheight', None)
    width = kwargs.pop('figwidth', None)
    ratio = kwargs.pop('figratio', 1.618)
    name = kwargs.pop('name', None)

    if 'nrows' in kwargs:
        ratio /= kwargs['nrows']

    if 'ncols' in kwargs:
        ratio *= kwargs['ncols']

    if 'figsize' not in kwargs:
        if height is not None and width is not None:
            kwargs['figsize'] = (width, height)
        elif height is not None:
            kwargs['figsize'] = (height * ratio, height)
        elif width is not None:
            kwargs['figsize'] = (width, width / ratio)
    if set(['nrows', 'ncols']).isdisjoint(kwargs.keys()):
        fig, axes = plt.subplots(nrows=1, **kwargs)
    else:
        fig, axes = plt.subplots(**kwargs)
        if "squeeze" not in kwargs and not hasattr(axes, "__getitem__"):
            axes = [axes]

    yield fig, axes
figure.__doc__ = figure.__doc__ + plt.subplots.__doc__
