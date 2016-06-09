import functools
import itertools
import multiprocessing
import operator
import pickle
import sys
import traceback


class ExceptionInSubprocess(Exception):
    def __init__(self, exception, traceback):
        super(ExceptionInSubprocess, self).__init__(self, exception)
        self.exception = exception
        self.traceback = traceback

    def __str__(self):
        if isinstance(self.exception, basestring):
            text = self.exception
        else:
            text = '{}: {}'.format(
                self.exception.__class__.__name__, self.exception)

        return '{}\n{}'.format(text, self.traceback)


def run_in_subprocess(fun):
    def call_to_pipe(p, *args, **kwargs):
        try:
            result = fun(*args, **kwargs)
            p.send((result, None))
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            exc_traceback = ''.join(traceback.format_tb(exc_traceback))
            exception = (exc_value, exc_traceback)
            try:
                # boost::python errors are not pickleable.
                pickle.dumps(exception)
            except pickle.PicklingError:
                exception = (str(exc_value), exc_traceback)
            p.send((None, exception))

    # Function needs to be at the module level in order to be callable
    # by multiprocessing.Process().
    call_to_pipe.__name__ = ('_{}_in_subprocess'.format(fun.__name__))
    setattr(sys.modules[__name__], call_to_pipe.__name__, call_to_pipe)

    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        receiving, sending = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(
            target=call_to_pipe, args=[sending] + list(args), kwargs=kwargs)
        p.start()
        result, exception = receiving.recv()
        p.join()

        if exception:
            exc_value, exc_traceback = exception
            raise ExceptionInSubprocess(exc_value, exc_traceback)

        return result
    return wrapper


def get_nested(params, *keys, **kwargs):
    keys = reduce(operator.add, (k.split(".") for k in keys), [])
    try:
        val = reduce(operator.getitem, keys, params)
        return val
    except KeyError:
        if 'default' in kwargs:
            return kwargs['default']
        raise


def set_nested(params, *args):
    keys, val = args[:-1], args[-1]
    keys = reduce(operator.add, (k.split(".") for k in keys), [])
    while keys:
        name = keys.pop(0)
        if keys:
            if name not in params:
                params[name] = {}
            params = params[name]
    prev, params[name] = params.get(name), val
    return prev


def partition(it, predicate=bool):
    it1, it2 = itertools.tee((predicate(val), val) for val in it)
    return ((val for pred, val in it1 if not pred),
            (val for pred, val in it2 if pred))


def chunked(it, n, fillvalue=None):
    args = [iter(it)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


class cached_property(object):
    """
    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """  # noqa

    def __init__(self, func):
        self.__doc__ = func.__doc__
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
