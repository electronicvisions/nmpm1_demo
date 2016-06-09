import os
from single_neuron import analysis, utils, plots

spans = analysis.Spans()
spans.add("base", 0.5e-4, 1.0e-4)
spans.add("exc", 1.5e-4, 2.5e-4)
spans.add("inh", 3.0e-4, 4.0e-4)

def plot(d, subsampling=200, xlim=(0, 8.0e-4)):
    label = lambda exp: os.path.basename(exp.directory)
    experiments = sorted(analysis.load_experiments(d), key=lambda exp: exp.ctime)
    if not experiments:
        return
    chunked = list(utils.chunked(experiments, 6))
    with plots.figure(nrows=len(chunked), figwidth=12) as (fig, axes):
        for ii, experiments in enumerate(chunked):
            ax = axes[ii]
            for exp in filter(None, experiments):
                ax.plot(exp.membrane[::subsampling, 0], exp.membrane[::subsampling, 1], label=label(exp))
            spans.annotate(ax, alpha=0.1)
            ax.legend(prop={'size':'small'}, loc='best')
            ax.set_xlabel('time [s]')
            ax.set_ylabel('membrane potential [V]')
            ax.set_xlim(*xlim)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        return fig, axes

if __name__ == "__main__":
    fig, _ = plot(".")
    fig.savefig("output.png")
