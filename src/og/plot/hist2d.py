from typing import Sequence, Literal

import ipdb
import matplotlib.pyplot as plt
import numpy as np


def hist2d(
    b_x: np.ndarray,
    b_y: np.ndarray,
    bins: int | Sequence[np.ndarray],
    ax: plt.Axes,
    b_val: np.ndarray | None = None,
    val_reduction: Literal["mean", "max", "min"] = None,
    **kwargs
):
    D = 2

    b_sample = np.stack([b_x, b_y], axis=1)

    nbin = np.empty(D, np.intp)
    edges = D * [None]

    for ii in range(D):
        if np.ndim(bins[ii]) == 0:
            if bins[ii] < 1:
                raise ValueError("`bins[{}]` must be positive, when an integer".format(ii))
            smin, smax = b_sample[:, ii].min(), b_sample[:, ii].max()
            try:
                n = bins[ii]

            except TypeError as e:
                raise TypeError("`bins[{}]` must be an integer, when a scalar".format(ii)) from e

            edges[ii] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[ii]) == 1:
            edges[ii] = np.asarray(bins[ii])
            if np.any(edges[ii][:-1] > edges[ii][1:]):
                raise ValueError("`bins[{}]` must be monotonically increasing, when an array".format(ii))
        else:
            raise ValueError("`bins[{}]` must be a scalar or 1d array".format(ii))

        nbin[ii] = len(edges[ii]) + 1  # includes an outlier on each end

    # Compute the bin number each sample falls into.
    # ( b_binidx_x, b_binidx_y )
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], b_sample[:, i], side="right")
        for i in range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for ii in range(D):
        # Find which points are on the rightmost edge.
        on_edge = b_sample[:, ii] == edges[ii][-1]
        # Shift these points one bin to the left.
        Ncount[ii][on_edge] -= 1

    # (b, ). Index to the flattened array. Original array has shape nbin.
    n_xy = np.ravel_multi_index(Ncount, nbin)

    minlength = nbin.prod()
    # Compute the number of repetitions in xy and assign it to the flattened histmat.
    n_hist = np.bincount(n_xy, weights=None, minlength=minlength)
    #    If we want to divide, don't divide by zero.
    n_hist_safe = np.where(n_hist > 0, n_hist, 1)

    if b_val is None:
        # We are computing the density.
        n_plotval = n_hist
    else:
        match val_reduction:
            case "mean":
                n_valsum = np.zeros(minlength, dtype=np.float32)
                np.add.at(n_valsum, n_xy, b_val)
                n_val = n_valsum / n_hist_safe
            case "max":
                n_valmax = np.full(minlength, -np.inf, dtype=b_val.dtype)
                np.maximum.at(n_valmax, n_xy, b_val)
                n_val = n_valmax
            case "min":
                n_valmin = np.full(minlength, np.inf, dtype=b_val.dtype)
                np.minimum.at(n_valmin, n_xy, b_val)
                n_val = n_valmin
            case _:
                raise ValueError("Invalid value for `val_reduction` {}".format(val_reduction))

        n_plotval = np.ma.array(n_val, mask=n_hist == 0)

    # Shape into a proper matrix
    bb_hist = n_plotval.reshape(nbin)

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D * (slice(1, -1),)
    bb_hist = bb_hist[core]

    # Don't plot zero density.
    bb_hist = np.ma.masked_less_equal(bb_hist, 0)

    xedges, yedges = edges
    pc = ax.pcolormesh(xedges, yedges, bb_hist.T, **kwargs)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    return pc
