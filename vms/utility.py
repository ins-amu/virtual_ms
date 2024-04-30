import os
import time
import tqdm
import torch
import logging
from os import stat
import numpy as np
import pylab as plt
import collections.abc
import multiprocessing as mp
from os.path import join
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr
from scipy.signal import find_peaks, peak_widths
from scipy.signal import filtfilt, butter, hilbert
from sbi.analysis.plot import _get_default_opts, _update, ensure_numpy # sbi 0.21.0
import sbi.utils as utils
from scipy.signal import detrend
from scipy.stats import zscore as _zscore
from torch.distributions import Uniform, Normal, HalfCauchy


def brute_sample(prior, num_samples, nx=None, ny=None, num_ensebmles=1):
    '''
    Args:
        prior (sbi.utils.torchutils.BoxUniform): prior information
        num_samples (int): number of samples to generate
        nx (int): number of samples in x dimension
        ny (int): number of samples in y dimension
        num_ensebmles (int): number of ensembles to generate

    Returns:
        ndarray: samples
    '''
    try:
        low = prior.base_dist.low.tolist()
        high = prior.base_dist.high.tolist()
    except:
        if isinstance(prior, utils.MultipleIndependent):
            low = [prior.dists[i].low.item() for i in range(2)]
            high = [prior.dists[i].high.item() for i in range(2)]
        else:
            raise ValueError("prior not supported")

    assert(len(high) == len(low))
    if len(high) == 1:
        # theta = torch.linspace(low[0], high[0], steps=num_samples)[:, None].float()
        theta = torch.linspace(low[0], high[0], steps=num_samples).float()
        return theta.repeat(1, num_ensebmles).T

    elif len(high) == 2:

        assert(nx is not None)
        assert(ny is not None)

        interval = np.abs([high[0] - low[0], high[1] - low[1]])

        'true' if True else 'false'
        step_x = interval[0] / (nx - 1) if (nx > 1) else interval[0]
        step_y = interval[1] / (ny - 1) if (ny > 1) else interval[1]

        theta = []
        for i in range(nx):
            for j in range(ny):
                theta.append([low[0] + i * step_x, low[1] + j * step_y])

        # for i in range(nx):
        #     for j in range(ny):
        #         x0 = [low[0] + i / nx * interval[0],
        #               low[1] + j / ny * interval[1]]
        #         theta.append(x0)

        theta = torch.tensor(theta).float()
        return theta.repeat(num_ensebmles, 1)


def timer(func):
    '''
    decorator to measure elapsed time

    Parameters
    -----------
    func: function
        function to be decorated
    '''

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        display_time(end-start, message="{:s}".format(func.__name__))
        return result
    return wrapper

def display_time(time, message=""):
    '''
    display elapsed time in hours, minutes, seconds

    Parameters
    -----------
    time: float
        elaspsed time in seconds
    '''

    hour = int(time/3600)
    minute = (int(time % 3600))//60
    second = time-(3600.*hour+60.*minute)
    print("{:s} Done in {:d} hours {:d} minutes {:09.6f} seconds".format(
        message, hour, minute, second))

def is_sequence(x):
    if isinstance(x, collections.abc.Sized):
        return True
    else:
        return False


def batch_file_PC(data_path, num_simulations, n_jobs, theta_filename):
    '''
    write batch file for parallel computing on PC
    '''

    job_filename = join(data_path, "log", f"script.sh")
    with open(job_filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"num_simulations={num_simulations}\n")
        f.write(f"n_jobs={n_jobs}\n")
        f.writelines(f"""
for i in $( seq 0 {num_simulations-1})
do
    if [ $(expr $i % $n_jobs) != "0" ]; then
        python3 -W ignore single_job.py {theta_filename} $i 0 &
    else
        wait
        python3 -W ignore single_job.py {theta_filename} $i 0 &
    fi
done
wait
"""
                     )
    return job_filename


def batch_file_JSC(data_path, sim_index, batch_step, n_jobs, theta_filename, jobname):

    job_filename = join(data_path, "log", f"script_{sim_index}.sh")
    with open(job_filename, "w") as f:

        # f.write("#!/bin/bash \n")
        f.write("#!/bin/bash -x \n")

        f.write("#SBATCH --account=icei-hbp-2021-0002\n")
        f.write("#SBATCH --time=24:00:00 \n")
        f.write("#SBATCH --nodes=1 \n")
        f.write("#SBATCH --partition=gpus\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.writelines(f"#SBATCH --job-name={jobname}-{sim_index}.job\n")
        f.writelines(f"#SBATCH --output={data_path}/log/{jobname}-{sim_index}.log\n")
        f.writelines(f"#SBATCH --error={data_path}/log/{jobname}-{sim_index}.err \n")

        # f.write("#SBATCH --nodes=1 \n")
        # f.write(f"#SBATCH --ntasks-per-node={n_jobs} \n")
        # f.write('export OMP_NUM_THREADS=1\n')
        # f.writelines("#SBATCH --ntasks-per-socket=1 \n")
        f.write(f"n_jobs={n_jobs}\n")
        f.writelines("""
module load Python
module load CUDA
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
                        """
                     )
        f.writelines(f"""
for i in $( seq {sim_index} {sim_index+batch_step-1})
do
    if [ $(expr $i % $n_jobs) != "0" ]; then
        python3 -W ignore single_job.py {theta_filename} $i 0 &
    else
        wait
        python3 -W ignore single_job.py {theta_filename} $i 0 &
    fi
done
wait
            """
                     )
    return job_filename


def get_limits(samples, limits=None):

    if type(samples) != list:
        samples = ensure_numpy(samples)
        samples = [samples]
    else:
        for i, sample_pack in enumerate(samples):
            samples[i] = ensure_numpy(samples[i])

    # Dimensionality of the problem.
    dim = samples[0].shape[1]

    if limits == [] or limits is None:
        limits = []
        for d in range(dim):
            min = +np.inf
            max = -np.inf
            for sample in samples:
                min_ = sample[:, d].min()
                min = min_ if min_ < min else min
                max_ = sample[:, d].max()
                max = max_ if max_ > max else max
            limits.append([min, max])
    else:
        if len(limits) == 1:
            limits = [limits[0] for _ in range(dim)]
        else:
            limits = limits
    limits = torch.as_tensor(limits)

    return limits


def posterior_peaks(samples, return_dict=False, **kwargs):

    opts = _get_default_opts()
    opts = _update(opts, kwargs)

    limits = get_limits(samples)
    samples = samples.numpy()
    n, dim = samples.shape

    try:
        labels = opts['labels']
    except:
        labels = range(dim)

    peaks = {}
    if labels is None:
        labels = range(dim)
    for i in range(dim):
        peaks[labels[i]] = 0

    for row in range(dim):
        density = gaussian_kde(
            samples[:, row],
            bw_method=opts["kde_diag"]["bw_method"])
        xs = np.linspace(
            limits[row, 0], limits[row, 1],
            opts["kde_diag"]["bins"])
        ys = density(xs)

        # y, x = np.histogram(samples[:, row], bins=bins)
        peaks[labels[row]] = xs[ys.argmax()]

    if return_dict:
        return peaks
    else:
        return list(peaks.values())


def get_dataset_path():
    import vms
    import os

    path = vms.__path__[0]
    path = join(path, "..", "dataset")
    if not os.path.isdir(path):
        raise ValueError("Dataset folder does not exist")
    return path
