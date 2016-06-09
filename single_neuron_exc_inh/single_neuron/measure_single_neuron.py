import os
import re
import sys

import numpy as np

from . import arguments
from . import utils

LOGFILE = "{}.log".format(__name__)


def setup_logger():
    """
    Has to be called separately in every process.
    """
    import argparse
    from pysthal.command_line_util import init_logger
    import pylogging

    logger = pylogging.get(__name__)

    init_logger("DEBUG", [
        (__name__, "TRACE"),
        ("halbe.fgwriter", "INFO"),
        ("hicann-system", "INFO"),
        ("Default", "INFO"),  # unset logger name, mostly hicann-system
    ])

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".")
    args, _ = parser.parse_known_args()

    pylogging.append_to_file(os.path.join(args.output_dir, LOGFILE))

    return logger


@utils.run_in_subprocess
def run_mapping(calib_dir, output_dir, wafer, hicann, skip_neurons, params):
    """
    :type hicann: HICANNOnWafer
    :param params: dictionary containing neuron parameters
    :param skip_neurons: number of non-functional dummy neurons to insert
    """

    from pymarocco import PyMarocco
    from pymarocco.results import Marocco
    from pymarocco.coordinates import BioNeuron
    import pyhmf as pynn
    import pysthal

    logger = setup_logger()

    marocco = PyMarocco()
    marocco.neuron_placement.default_neuron_size(
        utils.get_nested(params, "neuron.size", default=4))
    marocco.neuron_placement.restrict_rightmost_neuron_blocks(True)
    marocco.neuron_placement.minimize_number_of_sending_repeaters(False)
    marocco.backend = PyMarocco.None
    marocco.calib_backend = PyMarocco.XML
    marocco.calib_path = calib_dir
    marocco.param_trafo.use_big_capacitors = False
    marocco.persist = os.path.join(output_dir, "marocco.xml.gz")
    marocco.wafer_cfg = os.path.join(output_dir, "wafer_cfg.bin")
    marocco.default_wafer = wafer

    # FIXME: remove?
    marocco.param_trafo.alpha_v = 1000.0
    marocco.param_trafo.shift_v = 0.0

    pynn.setup(marocco=marocco)

    synaptic_input = {}
    for input_type, input_params in params["synaptic_input"].iteritems():
        if not utils.get_nested(input_params, "enabled", default=True):
            logger.info(
                "skipping disabled {!r} synaptic input".format(input_type))
            continue

        spike_times = utils.get_nested(
            input_params, "spike_times", default=None)
        if spike_times:
            start = spike_times["start"]
            stop = spike_times["stop"]
            step = spike_times["step"]
            spike_times = np.arange(start, stop, step)
            input_pop_model = pynn.SpikeSourceArray
            input_pop_params = {"spike_times": spike_times}
        else:
            raise NotImplementedError(
                "unknown config for {!r} synaptic input".format(input_type))

        logger.info(
            ("{!r} synaptic input will come from "
             "{} with parameters {!r}").format(
                input_type, input_pop_model.__name__, input_pop_params))
        synaptic_input[input_type] = pynn.Population(
            1, input_pop_model, input_pop_params)

    neuron_params = utils.get_nested(params, "neuron.parameters")
    neuron_model = getattr(pynn, utils.get_nested(
        params, "neuron.model", default="IF_cond_exp"))

    logger.info(
        "target population is {} neuron with parameters {!r}".format(
            neuron_model.__name__, neuron_params))

    # Force marocco to give us a different neuron by inserting
    # `Neuron_Number - 1` dummy neurons.
    populations = []
    for ii in range(0, skip_neurons + 1):
        populations.append(pynn.Population(
            1, neuron_model, neuron_params))
        marocco.manual_placement.on_hicann(populations[-1], hicann)
    target_pop = populations[-1]

    for input_type, input_pop in synaptic_input.iteritems():
        multiplicity = utils.get_nested(
            params, "synaptic_input", input_type, "multiplicity",
            default=1)
        assert multiplicity >= 1
        weight = utils.get_nested(
            params, "synaptic_input", input_type, "weight")
        con = pynn.AllToAllConnector(weights=weight)
        logger.info(
            ("connecting {!r} synaptic input "
             "to target population with weight {} "
             "via {} projections").format(
                 input_type, weight, multiplicity))
        for _ in xrange(multiplicity):
            pynn.Projection(input_pop, target_pop, con, target=input_type)

    pynn.run(params["duration"])
    pynn.end()

    wafer_cfg = pysthal.Wafer()
    wafer_cfg.load(marocco.wafer_cfg)
    results = Marocco.from_file(marocco.persist)
    return (BioNeuron(target_pop[0]), results, wafer_cfg)


def set_custom_dac_values(hicann, logical_neuron, wafer_cfg, params):
    """
    :type hicann: HICANNOnWafer
    """
    from pyhalbe import HICANN

    if "DAC" not in params:
        return wafer_cfg

    logger = setup_logger()

    # Overwrite analog neuron parameters with custom DAC values.
    for nrn in logical_neuron:
        logger.debug("Setting custom FG parameters for {}".format(
            nrn))
        for key in sorted(params["DAC"].keys()):
            value = params["DAC"][key]
            prev = wafer_cfg[hicann].floating_gates.getNeuron(
                nrn, getattr(HICANN.neuron_parameter, key))
            wafer_cfg[hicann].floating_gates.setNeuron(
                nrn, getattr(HICANN.neuron_parameter, key), value)
            if prev != value:
                logger.trace("{} changed from {} to {}".format(
                    key, prev, value))

    return wafer_cfg


def log_floating_gate_values(hicann, logical_neuron, wafer_cfg):
    from pyhalbe import HICANN

    logger = setup_logger()

    # Overwrite analog neuron parameters with custom DAC values.
    for nrn in logical_neuron:
        logger.debug("Initial FG parameters for {}:".format(
            nrn))
        for name in sorted(HICANN.neuron_parameter.names.keys()):
            if name.startswith('_'):
                continue
            param = getattr(HICANN.neuron_parameter, name)
            value = wafer_cfg[hicann].floating_gates.getNeuron(nrn, param)
            logger.debug("{!r}: {!r}".format(name, value))

    return wafer_cfg


def set_floating_gate_bias(hicann, wafer_cfg):
    """
    :type hicann: HICANNOnWafer
    """
    from pyhalbe import HICANN
    import Coordinate as C

    fgc = wafer_cfg[hicann].floating_gates
    for idx in map(C.Enum, xrange(fgc.getNoProgrammingPasses())):
        cfg = fgc.getFGConfig(idx)
        cfg.fg_biasn = 0
        cfg.fg_bias = 0
        fgc.setFGConfig(idx, cfg)

    return wafer_cfg


def record_membrane(
        output_dir, hicann, aout, placement_item, wafer_cfg,
        params
):
    """
    :type hicann: HICANNOnWafer
    :type aout: AnalogOnHICANN
    """
    import pysthal
    import Coordinate as C

    logical_neuron = placement_item.logical_neuron()
    hicann_cfg = wafer_cfg[hicann]
    hicann_cfg.enable_aout(logical_neuron.front(), aout)

    db = pysthal.MagicHardwareDatabase()
    wafer_cfg.connect(db)

    configurator = pysthal.HICANNv4Configurator()
    wafer_cfg.configure(configurator)

    adc = hicann_cfg.analogRecorder(aout)
    recording_time = params["duration"] / 1e3 / 1e4
    adc.setRecordingTime(recording_time)
    adc.activateTrigger()

    runner = pysthal.ExperimentRunner(recording_time)
    wafer_cfg.start(runner)

    if adc.hasTriggered():
        v = adc.trace()
        t = adc.getTimestamps()
        membrane = np.vstack((t, v)).T
    else:
        membrane = np.empty(0)

    address = placement_item.address()
    link = C.GbitLinkOnHICANN(address.toDNCMergerOnHICANN())
    spikes = hicann_cfg.receivedSpikes(link)
    if len(spikes):
        spikes = spikes[spikes[:, 1] == address.toL1Address()]

    return (membrane, spikes)


def neuron_directory(output_dir, logical_neuron):
    neuron = logical_neuron.front()
    basename = "hicann-{:03}-nrn-{:03}".format(
        neuron.toHICANNOnWafer().id().value(),
        neuron.toNeuronOnHICANN().id().value())
    index = 1
    while os.path.exists(os.path.join(output_dir, basename)):
        index += 1
        basename = basename.rsplit("_", 1)[0]
        basename = "{}_{}".format(basename, index)
    return os.path.join(output_dir, basename)


def add_options(parser):
    parser.add_argument(
        "--analog_out", choices=[0, 1], default=0,
        help="analog output to use")
    parser.add_argument(
        "--output-dir", default=".", metavar="DIR")
    parser.add_argument(
        "--calibration-dir", default=".", metavar="DIR",
        help="directiory containing calibration data")
    parser.add_argument(
        "--skip-neurons", type=int, metavar="COUNT", default=0,
        help="number of dummy neurons to insert")
    parser.add_argument(
        "--store-copy-of", type=str, metavar="FILE", default=[], nargs="+",
        help="store a copy of the provided file in the output directory")
    parser.add_argument(
        "--overwrite", type=arguments.key_value_param, metavar="KEY=VALUE",
        default=[], nargs="+",
        help="overwrite a parameter value")
    parser.add_argument(
        "--sweep", type=arguments.sweep_param, metavar="KEY=START:STOP[:STEP]",
        default=[], nargs="+",
        help="sweep a parameter value")
    parser.add_argument(
        "parameters",
        help="file containing the neuron parameters")


def main():
    import argparse
    import copy
    import gzip
    import pickle
    import shutil
    import tempfile
    import yaml

    import Coordinate as C

    from pysthal.command_line_util import add_default_coordinate_options
    from pysthal.command_line_util import add_logger_options

    parser = argparse.ArgumentParser(
        description="Single Neuron Measurement")
    add_default_coordinate_options(parser)
    add_logger_options(parser)
    add_options(parser)
    args = parser.parse_args()

    if not args.hicann or not args.wafer:
        parser.error("HICANN or wafer not specified")

    for key, _ in args.sweep:
        if not key.startswith("DAC."):
            parser.error("Sweeping only implemented for DAC values")

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except OSError:
            parser.error("Could not create output directory")

    logger = setup_logger()
    logger.info("Starting single neuron measurement")
    logger.debug(
        "Called with the following arguments:\n{}".format(sys.argv))

    temp_dir = tempfile.mkdtemp(dir=args.output_dir)
    logger.debug("Using temporary directory for output: {}".format(temp_dir))

    logger.info("Loading parameters from file {}".format(args.parameters))
    with open(args.parameters) as f:
        params = yaml.load(f)

    logger.debug("Loaded parameters:\n{}".format(params))

    for key, value in args.overwrite:
        prev = utils.set_nested(params, key, value)
        logger.debug(
            "Overwriting parameter {!r} with value {!r} (was {!r})".format(
                key, value, prev))

    for filename in args.store_copy_of:
        extra = ""
        if os.path.islink(filename):
            extra = ", which links to {}".format(os.readlink(filename))
        logger.info("Storing copy of {}{}".format(filename, extra))
        outname = os.path.join(temp_dir, os.path.basename(filename) + ".gz")
        with open(filename, "rb") as f_in, gzip.open(outname, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info("Starting marocco process")
    bio_neuron, results, wafer_cfg = run_mapping(
        args.calibration_dir, temp_dir, args.wafer,
        args.hicann, args.skip_neurons, params=params)

    items = list(results.placement.find(bio_neuron))
    assert len(items) == 1
    placement_item = items[0]
    logical_neuron = placement_item.logical_neuron()

    output_dir = neuron_directory(args.output_dir, logical_neuron)

    logger.debug("Moving temporary output directory to {}".format(output_dir))
    shutil.move(temp_dir, output_dir)

    log_floating_gate_values(args.hicann, logical_neuron, wafer_cfg)

    def measure(wafer_cfg, params, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = "logical_neuron.pickle"
        logger.info("Storing used neuron in {}".format(filename))
        with open(os.path.join(output_dir, filename), "wb") as f:
            pickle.dump(logical_neuron, f, protocol=-1)

        filename = "parameters.yaml.gz"
        logger.info("Storing used parameters in {}".format(filename))
        with gzip.open(os.path.join(output_dir, filename), "wb") as f:
            yaml.dump(params, f, default_flow_style=False)

        logger.info("Setting custom DAC values")
        wafer_cfg = set_custom_dac_values(
            args.hicann, logical_neuron, wafer_cfg, params)

        logger.info("Setting floating gate biases")
        wafer_cfg = set_floating_gate_bias(args.hicann, wafer_cfg)

        logger.info("Recording membrane voltage")
        membrane, spikes = record_membrane(
            args.output_dir, args.hicann, C.AnalogOnHICANN(args.analog_out),
            placement_item, wafer_cfg, params)

        logger.info("Saving results to directory {}".format(output_dir))
        np.save(os.path.join(output_dir, "membrane.npy"), membrane)
        np.save(os.path.join(output_dir, "spikes.npy"), spikes)

    if not args.sweep:
        measure(wafer_cfg, params, output_dir)
    else:
        logger.info("Beginning to sweep parameter ranges")

    for params_with_value, params_with_expr in arguments.unroll_sweep(args.sweep):
        label = ", ".join(map(lambda s: "=".join(map(str, s)),
                              params_with_value + params_with_expr))
        logger.info("Sweep {}".format(label))

        params_ = copy.deepcopy(params)
        for key, val in params_with_value:
            logger.debug("Setting parameter {!r} to {!r}".format(
                key, val))
            utils.set_nested(params_, key, val)
        for key, expr in params_with_expr:
            # We could support more complicated expressions here.
            val = utils.get_nested(params_, expr)
            logger.debug("Setting parameter {!r} to {!r} ({!r})".format(
                key, expr, val))
            utils.set_nested(params_, key, val)

        measure(wafer_cfg, params_, os.path.join(output_dir, label))

    logger.debug("Moving log file to output directory")
    shutil.move(os.path.join(args.output_dir, LOGFILE), output_dir)

    logger.info("Done.")
