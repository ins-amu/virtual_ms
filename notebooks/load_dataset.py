import h5py
import numpy as np
from copy import copy
import scipy.io as sio
from os.path import join
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter, hilbert
from mpl_toolkits.axes_grid1 import make_axes_locatable


class P_Dataset:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    def load_SC(self, group, index, normalize=True):
        ''' 
        load SC data from the given group and index

        parameters
        ------------
        group: str
            group name, options: control, patient
        index: int
            index of the subject, control: 0-19, patient: 0-17

        Returns
        ------------
        SC: ndarray
            structural connectivity matrix
        DL: ndarray
            distance matrix
        '''

        tracts = sio.loadmat(join(self.data_path, "tracts_DKT_25HC.mat"))
        if group == "control":
            assert(index < 20), "index should be less than 20"
            SC = tracts['tract_DKT_CTRL'][:, :, index]
            DL = tracts['tract_DKT_L_CTRL'][:, :, index]
        elif group == "patient":
            assert(index < 18), "index should be less than 18"
            SC = tracts['tract_DKT_MS'][:, :, index]
            DL = tracts['tract_DKT_L_MS'][:, :, index]

        if normalize:
            return self.normalize_SC(SC), DL
        else:
            return SC, DL
    
    def get_avg_SC(self, group, normalize=True):
        '''
        get average SC from the given group

        parameters
        ------------
        group: str
            group name, options: control, patient

        Returns
        ------------
        SC: ndarray
            structural connectivity matrix
        DL: ndarray
            distance matrix
        '''

        if group == "control":
            n_subj = 20
        elif group == "patient":
            n_subj = 18

        SC_list = []
        DL_list = []
        for i in range(n_subj):
            SC, DL = self.load_SC(group, i, normalize=normalize)
            SC_list.append(SC)
            DL_list.append(DL)

        return np.mean(SC_list, axis=0), np.mean(DL_list, axis=0)

    def load_TS(self, group, index):
        '''
        load signal from matfile

        parameters
        ----------
        group: str
            options: control, patient
        index: int
            index of subject
        fs: int
            sampling frequency
        Returns
        ----------
        signal: ndarray
            signal data [n_channels, n_samples]

        '''
        if group == "control":
            mat_filename = join(self.data_path, "ControlsMS.mat")
        else:
            mat_filename = join(self.data_path, "patientsMS.mat")

        with h5py.File(mat_filename, 'r') as f:
            _key = "ControlsMS" if (group == "control") else "patientsMS"
            ref = f[_key][index][0]
            res = np.array(f[ref])

        fs = 1024
        ns = res.shape[0]
        t = self.time_from_fs(fs, ns)

        return t, res.T

    @staticmethod
    def down_sample(t, ts, factor, axis=1, ftype="fir", **kwargs):
        '''
        downsample the x

        parameters
        ------------
        ts: ndarray
            signal data [n_channels, n_samples]
        fs: int
            sampling frequency
        factor: int
            downsampling factor
        '''

        ts_new = signal.decimate(ts, factor, axis=axis, ftype=ftype, **kwargs)
        nt = ts_new.shape[1]
        t_new = np.linspace(t[0], t[-1], nt, endpoint=False)
        return t_new, ts_new

    @staticmethod
    def resample(t, ts, num, axis=1, ftype="fir", **kwargs):
        '''
        resample the time series

        parameters
        ------------
        ts: ndarray
            signal data [n_channels, n_samples]
        num: int
            The number of samples in the resampled signal.
        '''

        ts_new = signal.resample(ts, num, axis=axis, **kwargs)
        nt = ts_new.shape[1]
        t_new = np.linspace(t[0], t[-1], nt, endpoint=False)
        return t_new, ts_new

    @staticmethod
    def time_from_fs(fs, n_samples):

        dt = 1.0 / fs
        T = dt * n_samples
        return np.linspace(0, T, n_samples, endpoint=False)

    @staticmethod
    def normalize_SC(SC):
        SC_ = copy(SC)
        np.fill_diagonal(SC_, 0.0)
        SC_ = SC_/np.max(SC_)
        SC_ = np.abs(SC_)
        assert(np.trace(SC_) == 0.0)
        return SC_

    @staticmethod
    def plot_SC(A, ax, title, xlabel=None, ylabel=None, cmap="jet", **kwargs):
        im = ax.imshow(A, interpolation='nearest', cmap=cmap, **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ax=ax)
        cbar.ax.tick_params(labelsize=15)
        ax.set_title(title, fontsize=16)
        if xlabel is not None:
            ax.set_ylabel(xlabel, fontsize=16.0)
        if ylabel is not None:
            ax.set_xlabel(ylabel, fontsize=16.0)
        ax.tick_params(labelsize=14)

    @staticmethod
    def plot_ts(t, ts, ax, ylabel="EEG", fontsize=14, color="teal"):
        ''' 
        plot signals

        parameters
        ----------
        ts: np.ndarray [n_channels, n_samples]
            signal to plot

        '''
        ax.plot(t, ts.T, lw=1, alpha=0.15, color=color)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(labelsize=13)
        ax.margins(x=0)

    @staticmethod
    def carpet_plot(t, ts, ax, ylabel='Region', fontsize=14, **kwargs):
        '''
        plot carpet plot

        parameters
        ----------
        t: np.ndarray [n_samples]
            time vector
        ts: np.ndarray [n_channels, n_samples]
            signal to plot

        '''
        nn = ts.shape[0]
        ax.plot(t, ts.T/np.max(ts)*2 + np.r_[:nn], **kwargs)
        ax.set_yticks(np.r_[::10])
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel("Time [s]", fontsize=fontsize)
        ax.set_yticks(np.arange(0, nn, 10))
        ax.margins(x=0, y=0)

    @staticmethod
    def spectral_power(ts, fs, **kwargs):
        '''!
        Calculate features from the spectral power of the given BOLD signal.

        parameters
        ----------
        ts: np.ndarray [n_channels, n_samples]
            signal time series
        fs: float
            sampling frequency [Hz]
        return np.ndarray
            spectral power [n_channels, num_freq_points]

        '''

        f, Pxx_den = signal.periodogram(ts, fs, **kwargs)
        return f, Pxx_den

    @staticmethod
    def welch(ts, fs, **kwargs):
        '''!
        Calculate features from the spectral power of the given BOLD signal.

        parameters
        ----------
        ts: np.ndarray [n_channels, n_samples]
            signal time series
        fs: float
            sampling frequency [Hz]
        return np.ndarray
            spectral power

        '''

        f, Pxx_den = signal.welch(ts, fs, **kwargs)
        return f, Pxx_den

    @staticmethod
    def plot_freq_spectrum(f, Pxx_den, ax, average=False, logscale="x", **kwargs):
        ''' 
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

        '''

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
        else:
            pass

        ax.set_ylabel('PSD [V**2/Hz]', fontsize=16)
        ax.set_xlabel("f [Hz]", fontsize=13)
        ax.tick_params(labelsize=13)
        ax.margins(x=0)

    @staticmethod
    def moving_average(x, w, axis=1, mode="same"):
        return np.apply_along_axis(lambda x: np.convolve(x, np.ones(w), mode) / w, axis=axis, arr=x)


def filter_butter_bandpass(sig, fs, lowcut, highcut, order=5):
    """
    Butterworth filtering function

    :param sig: [np.array] Time series to be filtered
    :param fs: [float] Frequency sampling in Hz
    :param lowcut: [float] Lower value for frequency to be passed in Hz
    :param highcut: [float] Higher value for frequency to be passed in Hz
    :param order: [int] The order of the filter.
    :return: [np.array] filtered frequncy
    """

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, sig)

def PSD_under_area(f, pxx, opt=None):

    avg = opt['average_over_channels']
    normalize = opt['normalize']

    fmin = opt['fmin']
    fmax = opt['fmax']

    if normalize:
        pxx = pxx/pxx.max()

    idx = np.logical_and(f >= fmin, f <= fmax).tolist()
    if len(idx) > 0:
        if avg:
            # pxx = np.mean(pxx, axis=0)
            area = np.trapz(pxx[idx], f[idx])
            return area

        else:
            area = np.trapz(pxx[:, idx], f[idx], axis=1).reshape(-1)
            return area
    else:
        return [np.nan]
