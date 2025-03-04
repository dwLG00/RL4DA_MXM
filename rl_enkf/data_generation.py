import numpy as np
import pandas as pd
from l96 import L96
from enkf import eakf
from tqdm import tqdm

def generate_eakf(l96_args=(40, 8, 3600, 0.1), initial_condition=None, ensemble_condition=None, Nens=20, noise=0.1):
    N, F, timesteps, dt = l96_args
    system = L96(N, F)

    H = np.eye(N)
    R = np.eye(N)
    ground_truth = initial_condition()
    ensembles = [ensemble_condition() for _ in range(Nens)]

    l96_data = generate_l96(N, F, timesteps, dt)
    t = 0

    posteriors = []
    observation_differences = []
    for i in tqdm(range(timesteps)):
        priors = [runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in ensembles]
        prior_mean = np.mean(priors)
        obs = H @ l96_data[i, :]
        error = obs - prior_mean
        stacked_ensembles = np.stack(ensembles, 1)
        new_posteriors = eakf(Nens, N, stacked_ensembles, H, noise, False, None, obs)
        ensemble_concat = np.concat(ensembles)
        posteriors.append(ensemble_concat)
        ensembles = np.unstack(new_posteriors, axis=1)
        observation_differences.append(error)

    return l96_data, posteriors, observation_differences

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
    import inspect
    N = 40
    F = 8
    timesteps = 3600 * 20
    dt = 0.05
    noise = (0, 0.1)
    #data = generate_l96(N, F, timesteps, dt)
    initial_condition = lambda: np.ones(N, dtype=np.float32) + np.random.multivariate_normal(noise[0] * np.ones(N), noise[1] * np.eye(N))
    data, posteriors, obs_differences = generate_eakf(l96_args=(N, F, timesteps, dt),
        initial_condition=initial_condition, ensemble_condition=initial_condition,
        Nens=20, noise=0.1)

    posteriors = np.array(posteriors)
    obs_differences = np.array(obs_differences)
    print('shapes: data %s, posts %s, obsdiffs %s' % (data.shape, posteriors.shape, obs_differences.shape))
    concatted = np.concatenate([data, posteriors, obs_differences], axis=1)
    df = pd.DataFrame(concatted)
    df.to_csv(f'./data/lorenz96_N{N}_F{F}_enkf.csv')
    with open(f'./data/lorenz96_N{N}_F{F}_enkf.txt', 'w') as f:
        f.write(f'Shape: stack[data: {data.shape}, posteriors: {data.shape}, obsdiff: {obs_differences.shape}]\n')
        f.write(f'Parameters: N={N}, F={F}, timesteps={timesteps}, dt={dt}\n')
        f.write(f'Initial condition: noise mean {noise[0]}, noise variance {noise[1]}, base value np.ones(N, dtype=float32)')
        #initial_condition_string = ' = '.join(str(inspect.getsourcelines(initial_condition)[0]).strip("['\\n']").split(" = ")[1:]) # don't question it
        #f.write('Initial condition: `%s`' % initial_condition)
