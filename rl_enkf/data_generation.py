import numpy as np
import pandas as pd
from l96 import L96
from enkf import eakf
from tqdm import tqdm

def generate_eakf(**kwargs):
    N, F, timesteps, dt = kwargs.get('l96_args')
    system = L96(N, F)
    initial_condition = kwargs.get('initial_condition')
    ensemble_condition = kwargs.get('ensemble_condition')
    Nens = kwargs.get('Nens', N // 2)
    noise = kwargs.get('noise', 0.1)

    H = np.eye(N)
    R = np.eye(N)
    ground_truth = initial_condition()
    ensembles = [ensemble_condition() for _ in range(Nens)]

    l96_data = generate_l96(N, F, timesteps, dt)
    t = 0

    priors = []
    observation_differences = []
    for i in tqdm(range(timesteps - 1)):
        posteriors = (runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in ensembles)
        posterior_mean = mean(posteriors)
        obs = H @ l96_data[i, :]
        obs_diff = obs - posterior_mean
        new_priors = eakf(Nens, N, np.stack(ensembles, 1), H, noise, False, None, obs)
        ensemble_concat = np.concat(ensembles)
        priors.append(ensembles_concat)
        observation_differences.append(new_priors)
        ensembles = np.unstack(new_priors)

    return l96_data, priors, observation_differences

def generate_l96(N, F, timesteps, dt):
    '''Generate l96 data using runge-kutta approx.
    Initial conditions will be initialized to [F, ..., F] + N(0, 0.01)
    '''
    system = L96(N, F)
    u0 = np.ones(N) * F + np.random.multivariate_normal(np.zeros(N), np.eye(N) * 0.01)
    data = [u0]
    t = 0

    for i in tqdm(range(timesteps - 1)):
        u = data[-1]
        u = runge_kutta_4(system.dx, u, t, dt)
        data.append(u)
        t += dt

    return np.array(data)

def runge_kutta_4(func, x0, t, dt):
    '''Apply runge-kutta to a function'''
    k1 = func(x0, t)
    k2 = func(x0 + (dt / 2.0) * k1, t + (dt / 2.0))
    k3 = func(x0 + (dt / 2.0) * k2, t + (dt / 2.0))
    k4 = func(x0 + dt * k3, t + dt)

    return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

if __name__ == '__main__':
    N = 40
    F = 8
    timesteps = 3600 * 20
    dt = 0.05
    data = generate_l96(N, F, timesteps, dt)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f'./data/lorenz96_N{N}_F{F}.csv')
