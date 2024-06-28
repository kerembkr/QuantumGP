import matplotlib.pyplot as plt
from src.utils.utils import save_fig
from matplotlib.ticker import MaxNLocator


def plot_fast_slow(cost_history, epochs, epochs_bo, opt_name, iters):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    plt.plot(cost_history, "grey", linewidth=1.5)
    plt.scatter(range(epochs_bo, epochs_bo + iters), cost_history[epochs_bo:], c="r", linewidth=1,
                label=opt_name)
    plt.scatter(range(epochs_bo), cost_history[0:epochs_bo], c="g", linewidth=1, label="Bayesian Optimization")

    # plot minimum bayes-opt value
    min_value = min(cost_history[0:epochs_bo])
    min_index = cost_history[0:epochs_bo].index(min_value)
    plt.scatter(min_index, min_value, c="y", linewidth=1, label="Best Guess")

    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel("Cost Function", fontsize=15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(direction="in", labelsize=12, length=10, width=0.8, colors='k')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.legend()

    # Save the figure as PNG
    save_fig(opt_name + '.png')


def plot_costs(data, save_png=False, title=None, log=False, fname=None):
    # plot curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for label, cost in data.items():
        ax.plot(cost, linewidth=2.0, label=label)
    if log:  # logarithmic
        ax.set_yscale('log', base=10)
    ax.set_xlabel("Iteration", fontsize=18, labelpad=15, fontname='serif')
    ax.set_ylabel("Cost Function", fontsize=18, labelpad=15, fontname='serif')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(direction="in", labelsize=12, length=10, width=0.8, colors='k')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.legend()
    legend = ax.legend(frameon=True, fontsize=12)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.2)
    if title is not None:
        ax.set_title(title, fontsize=18, fontname='serif')

    if save_png:
        # save_fig(fname)
        save_fig(["vqls/", fname])

