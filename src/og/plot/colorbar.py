import matplotlib.pyplot as plt


def colorbar(mappable, ax: plt.Axes, use_gridspec=True, **kwargs):
    fig: plt.Figure = ax.get_figure()
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    return fig.colorbar(mappable, cax=cax, use_gridspec=use_gridspec, **kwargs)
