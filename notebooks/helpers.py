import os
import json
import torch
import numpy as np
import pylab as plt
from numba import jit
import seaborn as sns
from copy import copy
from scipy import signal
from os.path import join
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


def normalize_theta(theta, a=[800.0, 2.5], b=[1800.0, 30.0]):
    '''
    Normalize the parameters to the range [0, 1]

    Parameters
    ----------
    theta : torch.tensor
        Parameters to be normalized
    a : list
        Lower bound of the parameters
    b : list
        Upper bound of the parameters

    Returns
    -------
    theta_norm : torch.tensor
        Normalized parameters
    '''
    theta_norm = torch.zeros(theta.shape)
    for i in range(theta.shape[1]):
        theta_norm[:, i] = (theta[:, i] - a[i]) / (b[i] - a[i])
    return theta_norm

def denormalize_theta(theta, a=[800.0, 2.5], b=[1800.0, 30.0]):
    '''
    Denormalize the parameters to the range [a, b]

    Parameters
    ----------
    theta : torch.tensor
        Parameters to be denormalized
    a : list
        Lower bound of the parameters
    b : list
        Upper bound of the parameters

    Returns
    -------
    theta_denorm : torch.tensor
        Denormalized parameters
    '''
    theta_denorm = torch.zeros(theta.shape)
    for i in range(theta.shape[1]):
        theta_denorm[:, i] = theta[:, i] * (b[i] - a[i]) + a[i]
    return theta_denorm


def joint_dist(ax, samples, limits=[(0., 1.), (0.0, 1.0)]):
    density = gaussian_kde(samples[:, [1, 0]].T, bw_method='scott')
    col = 1
    row = 0
    X, Y = np.meshgrid(
        np.linspace(limits[col][0], limits[col][1], 50,),
        np.linspace(limits[row][0], limits[row][1], 50))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(density(positions).T, X.shape)
    ax.imshow(Z, extent=(limits[col][0], limits[col][1],
                         limits[row][0], limits[row][1]),
              origin="lower",
              aspect="auto")
    return ax



def plot_ts(t, y, figsize=(15, 3.5), step=1, **kwargs):
    """
    plot time series

    Parameters
    ----------
    t : array
        time vector
    y : array
        time series [nnodes, ntime]
    """

    N = y.shape[0]
    fig, ax = plt.subplots(1, figsize=figsize, sharex=True)
    ax.plot(t, y[:N, ::step].T, **kwargs)

    ax.set_ylabel("Real Z", fontsize=16)
    ax.set_xlabel("time [s]", fontsize=16)
    ax.margins(x=0)
    ax.tick_params(labelsize=14)

    return fig, ax


def plot_freq_spectrum(f, Pxx_den, ax, average=False, logscale="x", **kwargs):
    """
    plot frequency spectrum

    parameters
    ----------
    f: np.ndarray [n_freq_points]
    Pxx_den : np.ndarray [n_channels, n_freq_points]
        frequency spectrum
    ax: matplotlib.axes
        axis to plot on
    logscale: str
        logscale of axis (default: x)
        options: x, y, xy

    """

    y = Pxx_den.T
    if average:
        y = np.mean(y, axis=1)

    ax.plot(f, y, **kwargs)
    if logscale == "x":
        ax.set_xscale("log")
    elif logscale == "y":
        ax.set_yscale("log")
    elif logscale == "xy":
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_ylabel("PSD [V**2/Hz]", fontsize=16)
    ax.set_xlabel("f [Hz]", fontsize=13)
    ax.tick_params(labelsize=13)
    ax.margins(x=0)
