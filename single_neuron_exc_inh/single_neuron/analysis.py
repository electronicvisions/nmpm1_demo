import collections
import gzip
import os
import pickle
import time
import yaml

try:
    import uncertainties
except ImportError:
    uncertainties = None

import numpy as np

from pyhalbe.Coordinate import *
from pymarocco.coordinates import LogicalNeuron

from . import utils


class Experiment(object):
    def __init__(self, directory):
        assert os.path.isdir(directory)
        self.directory = os.path.abspath(directory)
        for filename in ["logical_neuron.pickle",
                         "parameters.yaml.gz",
                         "membrane.npy"]:
            assert os.path.exists(self.relative(filename))

    def clear(self):
        directory = self.directory
        self.__dict__.clear()
        self.directory = directory

    def relative(self, filename):
        return os.path.join(self.directory, filename)

    @utils.cached_property
    def ctime(self):
        return os.path.getctime(self.directory)

    def strftime(self, format):
        return time.strftime("%F %H:%M:%S", time.localtime(self.ctime))

    @utils.cached_property
    def parameters(self):
        with gzip.open(self.relative("parameters.yaml.gz")) as f:
            return yaml.load(f)

    @utils.cached_property
    def neuron(self):
        with open(self.relative("logical_neuron.pickle"), "rb") as f:
            return pickle.load(f)

    @utils.cached_property
    def membrane(self):
        return np.load(self.relative("membrane.npy"))


def load_experiments(d):
    experiments = []
    for dirpath, dirnames, filenames in os.walk(os.path.abspath(d)):
        if 'membrane.npy' not in filenames:
            continue
        exp = Experiment(dirpath)
        experiments.append(exp)
    return experiments


class Span(collections.namedtuple("Span", "start stop label")):
    __slots__ = ()

    def slice(self, data, axis=None):
        x, y = data.T
        result = []
        start_idx = np.argmax(x > self.start)
        stop_idx = np.argmin(x < self.stop)
        return data[
            start_idx:stop_idx,
            axis if axis is not None else slice(None)]

    def mean(self, data):
        data = self.slice(data, axis=1)
        mean = np.mean(data)
        if uncertainties:
            std = np.std(data)
            return uncertainties.ufloat(mean, std)
        return mean

    def annotate(self, ax, **kwargs):
        return ax.axvspan(self.start, self.stop, **kwargs)


class Spans(object):
    def __init__(self):
        self.spans = {}

    def add(self, label, ymin, ymax):
        assert ymax > ymin
        self.spans[label] = Span(ymin, ymax, label)

    def annotate(self, ax, **kwargs):
        for span in self.spans.values():
            span.annotate(ax, **kwargs)

    def __getitem__(self, label):
        return self.spans[label]
