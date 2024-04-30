import os
import pickle
import torch
import numpy as np
from time import time
from os.path import join
import sbi.utils as utils
from vms.utility import timer
from vms.utility import display_time
from sbi.inference import SNPE


class Inference:
    def __init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    @timer
    def train(self,
              num_simulations,
              prior,
              x,
              theta,
              num_threads=1,
              method="SNPE",
              device="cpu",
              density_estimator="maf"
              ):
        
        '''!
        train the Neural Network.


        @param prior torch.distributions.Distribution
            prior distribution.
        @param x torch.Tensor
            feature data.
        @param theta torch.Tensor
            parameters.
        @param num_threads int
            number of threads to use.
        @param method str
            method to use for training.
        @param device str
            Training device, e.g., "cpu", "cuda: or "cuda:{0, 1, ...}".
        @param density_estimator  str
            density estimator to use for training. one of (nsf, maf, mdn, made).
        \return posterior 
            posterior distribution.

        '''

        torch.set_num_threads(num_threads)

        # self._device = process_device(device, prior=prior)

        if (len(x.shape) == 1):
            x = x[:, None]
        if (len(theta.shape) == 1):
            theta = theta[:, None]

        x = x[:num_simulations, :]
        theta = theta[:num_simulations, :]

        if method == "SNPE":
            inference = SNPE(
                prior=prior, density_estimator=density_estimator, device=device)
        elif method == "SNLE":
            inference = SNLE(
                prior=prior, density_estimator=density_estimator, device=device)
        elif method == "SNRE":
            inference = SNRE(
                prior=prior, density_estimator=density_estimator, device=device)
        else:
            print("unknown method, choose SNLE, SNRE or SNPE.")
            exit(0)

        # inference._device = self._device
        inference = inference.append_simulations(theta, x)
        _density_estimator = inference.train()
        posterior = inference.build_posterior(_density_estimator)

        return posterior
    # -------------------------------------------------------------------------

    def sample_posterior(self,
                         obs_stats,
                         num_samples,
                         posterior):
        '''!
        sample from the posterior using the given observation statistics.

        \param obs_stats torch.tensor
            observation statistics
        \param num_samples int  
            number of samples
        \param posterior torch.tensor
            posterior
        '''

        samples = posterior.sample((num_samples,), x=obs_stats)
        return samples
