from typing import overload

import matplotlib.pyplot as plt


class EzAxes:
    def __init__(self, axes: list[plt.Axes]):
        self.axes = axes
        self.idx = 0

    @property
    def __len__(self):
        return len(self.axes)

    @overload
    def get(self) -> plt.Axes: ...

    @overload
    def get(self, n_axes: int) -> list[plt.Axes]: ...

    def get(self, n_axes: int | None = None):
        if n_axes is None:
            ax = self.axes[self.idx]
            self.idx += 1
            if self.idx >= len(self.axes):
                raise IndexError("No more axes available.")

            return ax

        axes = self.axes[self.idx : self.idx + n_axes]
        self.idx += n_axes
        if self.idx >= len(self.axes):
            raise IndexError("No more axes available.")
        return axes
