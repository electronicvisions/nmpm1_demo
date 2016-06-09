import argparse
import re
import itertools
import yaml


from . import utils


RE_SWEEP = re.compile("(-?\d+):(-?\d+)(?::(-?\d+))?")


def key_value_param(arg):
    """
    Parameter of the type `KEY=VALUE`, where `VALUE` can be any valid
    yaml epression.
    """
    try:
        key, val = arg.split("=", 1)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "use colon to separate key from value in parameter option")

    return key, yaml.load(val)


def sweep_param(arg):
    """
    Parameter of the type `KEY=START:STOP[:STEP]`, indicating a sweep
    of an integer option.  A second form `KEY=EXPR` is allowed to
    indicate that the value should be computed based on the value of
    other sweeped parameters.
    """
    key, rest = arg.split("=", 1)

    if ":" in rest:
        m = RE_SWEEP.match(rest)
        if not m:
            raise argparse.ArgumentTypeError(
                "could not parse sweep parameter {!r}".format(arg))
        start = int(m.group(1), 10)
        stop = int(m.group(2), 10)
        step = int(m.group(3), 10) if m.group(3) else None
        val = slice(start, stop, step)
    elif "," in rest:
        # rest is list of single numeric values
        val = [int(n, 10) for n in rest.split(",")]
    else:
        try:
            # rest is a single number
            val = int(rest, 10)
        except ValueError:
            # rest is expression/name of parameter to copy
            val = rest

    return key, val


def iter_from_values(values):
    if isinstance(values, slice):
        if values.step:
            return xrange(values.start, values.stop, values.step)
        return xrange(values.start, values.stop)
    elif isinstance(values, list):
        return values
    return [values]


def unroll_sweep(parameters):
    """
    Generate sets of parameter values from sweep instructions.
    For each combination
    """
    if not parameters:
        return

    is_expr = lambda item: isinstance(item[1], basestring)
    with_values, with_expr = utils.partition(parameters, is_expr)
    with_values = list(with_values)
    with_expr = list(with_expr)

    keys = [key for key, _ in with_values]
    slices = [iter_from_values(val) for _, val in with_values]

    for vals in itertools.product(*slices):
        yield zip(keys, vals), with_expr
