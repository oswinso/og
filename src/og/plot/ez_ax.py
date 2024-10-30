from typing import overload

import matplotlib.pyplot as plt
from jaxtyping import Shaped


class EzAxes:
    """Convenience for working with multiple axes."""

    def __init__(self, axes: Shaped[plt.Axes, "ncol"]):
        assert len(axes) > 0 and isinstance(axes[0], plt.Axes)
        self.axes: list[plt.Axes] = list(axes)
        self.fig = axes[0].figure
        # Counter denoting the next unused axis.
        self.ax_cntr = 0
        self.size = len(axes)

    @overload
    def get(self) -> plt.Axes: ...

    @overload
    def get(self, num: int) -> list[plt.Axes]: ...

    def get(self, num: int | None = None):
        if num is None:
            # Get a singular axis.
            cntr_new = self.ax_cntr + 1
            if cntr_new > self.size:
                raise ValueError("No more axes available.")

            ax = self.axes[self.ax_cntr]
            self.ax_cntr = cntr_new
            return ax

        # Return num number of axes.
        cntr_new = self.ax_cntr + num
        if cntr_new > self.size:
            raise ValueError("No more axes available.")

        axes = self.axes[self.ax_cntr : cntr_new]
        self.ax_cntr = cntr_new
        return axes

    def __getitem__(self, idx):
        return self.axes.__getitem__(idx)
