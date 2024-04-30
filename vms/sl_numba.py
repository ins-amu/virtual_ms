import numpy as np
from numba import njit
from numpy.random import rand, randn


def timer(func):
    '''
    Timer decorator
    '''
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end-start:.2f} seconds")
        return result
    return wrapper


# @timer
@njit(nogil=True)
def intg_sl(W, tr_len, tspan, dt, a, sigma, G, velocity, freq=40, decimate=1, t_cut=0):
    '''
    integrate the SL model
    '''

    nstep = int(np.floor((tspan[1] - tspan[0]) / dt))
    ncut = int(np.floor(t_cut / dt))
    t = np.linspace(tspan[0], tspan[1], nstep)
    nn = W.shape[0]
    omega = 2 * np.pi * freq

    delays = np.floor(tr_len / velocity).astype(np.int64)

    maxdelay = np.max(delays)

    X = np.zeros((nn, int(np.floor((nstep-ncut)/decimate))), dtype=np.complex_)
    state = 1e-5 * rand(nn, maxdelay+1) + 1e-5 * 1J * rand(nn, maxdelay+1)

    store_count = 0
    for it in range(nstep):

        curr_t = it * dt
        curr_z = np.reshape(state[:, -1].copy(), (nn, 1))

        state_z = state.copy()
        dly_state = np.zeros((nn, nn), dtype=np.complex_)

        for i in range(nn):
            for j in range(nn):
                if W[i, j] > 0:
                    dly_state[i, j] = state_z[j, - (delays[i, j]+1)] - state_z[i, -1]

        input_dly = np.reshape(
            np.sum(np.multiply(W, dly_state), axis=1), (nn, 1))
        dz = curr_z * (a + (omega * 1J) - np.abs(curr_z**2)) + \
            G * input_dly + sigma*(randn(nn, 1) + 1J*randn(nn, 1))

        new_z = curr_z + dt * dz
        state[:, :-1] = state[:, 1:]
        state[:, -1] = np.reshape(new_z, (nn))

        if curr_t >= (t_cut):
            if (it % decimate) == 0:
                X[:, store_count] = np.reshape(new_z, (nn))
                store_count += 1
    return np.real(X), t[t >= t_cut][::decimate]
