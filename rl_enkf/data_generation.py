import numpy as np

from l96 import L96
from enkf import eakf
from tqdm import tqdm
from construct_gc import construct_GC
import matplotlib.pyplot as plt

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

def rmse(a, b):
    N = a.shape[0]
    return np.sqrt(np.linalg.norm(a - b) / N)

def mae(a, b):
    return np.mean(np.abs(a - b))

def _generate_eakf(l96_args=(40, 8, 100, 0.01, 100), initial_condition=None, ensemble_condition=None, Nens=20, noise=0.1, inflation_coef=1.1, distance=rmse, model=None):
    N, F, obs_freq, dt, timesteps = l96_args
    system = L96(N, F)

    H = np.eye(N)
    R = np.eye(N) * noise
    ground_truth = initial_condition() # (N,) array of initial points
    ensembles = [ensemble_condition(i) for i in range(Nens)] # (N, Nens) array, this is the ensemble around each truth
    l96_data = generate_l96(N, F, timesteps * obs_freq + 1, dt) # generate trailing, as we don't use the first data point
    t = 0

    CMat = construct_GC(3, N, np.arange(0, N))

    ground_truth, priors_array, posteriors_array = [], [], []
    for i in tqdm(range(1, obs_freq)):
        priors = ensembles[:]
        for _ in range(timesteps):
            priors = [runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in priors]
            t += dt
        prior_mean = sum(priors) / Nens # prior mean
        gt = l96_data[i * timesteps, :] # ground truth
        ground_truth.append(gt)
        priors_array.append(prior_mean)
        #background_error.append(distance(gt, prior_mean)) # forecast vs ground truth

        obs = H @ gt + np.random.multivariate_normal(np.zeros(N), R) # get observation
        prior_stack = np.stack(priors, 1)
        prior_mean_stack = np.stack([prior_mean] * Nens, 1)
        inflated_priors = prior_mean_stack + np.sqrt(inflation_coef) * (prior_stack - prior_mean_stack) # inflate priorso

        new_posteriors = eakf(Nens, N, inflated_priors, H, noise, 1, CMat, obs) # get posterior distribution
        posterior_mean = np.mean(new_posteriors, axis=1)
        #analysis_error.append(distance(gt, posterior_mean))
        posteriors_array.append(posterior_mean)

        ensembles = np.unstack(new_posteriors, axis=1)
        #ensembles = np.moveaxis(new_posteriors, 1, 0) # my version of numpy is outdated

    return np.array(ground_truth), np.array(priors_array), np.array(posteriors_array)


def generate_eakf(l96_args=(40, 8, 100, 0.01, 100), initial_condition=None, ensemble_condition=None, Nens=20, noise=0.1, inflation_coef=1.1, distance=rmse, model=None):
    N, F, obs_freq, dt, timesteps = l96_args
    system = L96(N, F)

    H = np.eye(N)
    R = np.eye(N) * noise
    ground_truth = initial_condition() # (N,) array of initial points
    ensembles = [ensemble_condition(i) for i in range(Nens)] # (N, Nens) array, this is the ensemble around each truth
    l96_data = generate_l96(N, F, timesteps * obs_freq + 1, dt) # generate trailing, as we don't use the first data point
    t = 0

    CMat = construct_GC(3, N, np.arange(0, N))

    background_error, analysis_error = [], []
    for i in tqdm(range(1, obs_freq)):
        priors = ensembles[:]
        for _ in range(timesteps):
            priors = [runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in priors]
            t += dt
        prior_mean = np.mean(priors) # prior mean
        gt = l96_data[i * timesteps, :] # ground truth
        background_error.append(distance(gt, prior_mean)) # forecast vs ground truth

        obs = H @ gt + np.random.multivariate_normal(np.zeros(N), R) # get observation
        prior_stack = np.stack(priors, 1)
        inflated_priors = prior_mean + np.sqrt(inflation_coef) * (prior_stack - prior_mean) # inflate priorso

        new_posteriors = eakf(Nens, N, inflated_priors, H, noise, 1, CMat, obs) # get posterior distribution
        posterior_mean = np.mean(new_posteriors, axis=1)
        analysis_error.append(distance(gt, posterior_mean))

        ensembles = np.unstack(new_posteriors, axis=1)
        #ensembles = np.moveaxis(new_posteriors, 1, 0) # my version of numpy is outdated

    return l96_data[0::timesteps][1:], background_error, analysis_error

def generate_training_data(l96_args=(40, 8, 100, 0.01, 100), initial_condition=None, ensemble_condition=None, Nens=20, noise=0.1, localization_coef=3, inflation_coef=1.1):
    N, F, obs_freq, dt, timesteps = l96_args
    system = L96(N, F)

    H = np.eye(N)
    R = np.eye(N) * noise
    ground_truth = initial_condition() # (N,) array of initial points
    ensembles = [ensemble_condition(i) for i in range(Nens)] # (N, Nens) array, this is the ensemble around each truth
    initial_ensembles = [array.copy() for array in ensembles] # deep copy, we will need this later
    l96_data = generate_l96(N, F, timesteps * obs_freq + 1, dt) # generate trailing, as we don't use the first data point
    t = 0
    CMat = construct_GC(localization_coef, N, np.arange(0, N))

    observations = []
    ensembles_data = [[] for _ in range(Nens)]
    for i in tqdm(range(obs_freq)):
        priors = ensembles[:]
        for _ in range(timesteps):
            priors = [runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in priors]
            t += dt
        prior_mean = np.mean(priors) # prior mean
        gt = l96_data[i * timesteps, :] # ground truth

        obs = H @ gt + np.random.multivariate_normal(np.zeros(N), R) # get observation
        observations.append(obs)
        prior_stack = np.stack(priors, 1)
        inflated_priors = prior_mean + np.sqrt(inflation_coef) * (prior_stack - prior_mean) # inflate priorso

        new_posteriors = eakf(Nens, N, inflated_priors, H, noise, 1, CMat, obs) # get posterior distribution
        posterior_mean = np.mean(new_posteriors, axis=1)

        ensembles = np.unstack(new_posteriors, axis=1)
        for idx in range(Nens):
            ensembles_data[idx].append(ensembles[idx])
        #ensembles = np.moveaxis(new_posteriors, 1, 0) # my version of numpy is outdated

    return np.array(observations), [np.array(ensemble) for ensemble in ensembles_data], initial_ensembles

def snapshot_figure():
    N = 40
    np.random.seed(0)
    initial_condition = lambda: np.ones(N) + np.random.multivariate_normal(np.zeros(N), 0.01 * np.eye(N))
    ensemble_condition = lambda i: initial_condition()

    ground_truth, priors, posteriors = _generate_eakf(
        l96_args=(40, 8, 100, 0.01, 100),
        initial_condition=initial_condition,
        ensemble_condition=ensemble_condition,
        inflation_coef=3
    )

    gt = ground_truth[-1, :]
    prior = priors[-1, :]
    posterior = posteriors[-1, :]

    theta = 2 * np.pi * np.arange(0, 1, 1/40)
    #prior_diff = abs(gt - prior)
    #posterior_diff = abs(gt - posterior)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    line1 = ax.plot(theta, gt, label="ground truth")
    line2 = ax.plot(theta, prior, label="prior")
    line3 = ax.plot(theta, posterior, label="posterior")
    ax.grid(True)
    ax.legend()
    ax.set_title('Ground Truth vs Prior vs Posterior L96')

    plt.show()


def error_figure():
    N = 40
    np.random.seed(0)
    initial_condition = lambda: np.ones(N) + np.random.multivariate_normal(np.zeros(N), 0.01 * np.eye(N))
    ensemble_condition = lambda i: initial_condition()

    data, background_error, analysis_error = generate_eakf(
        l96_args=(40, 8, 100, 0.01, 100),
        initial_condition=initial_condition,
        ensemble_condition=initial_condition,
        inflation_coef=3,
        distance=mae
    )

    length = len(data)
    background_error = np.array(background_error)
    analysis_error = np.array(analysis_error)

    print('shapes: data %s, background_error %s, analysis_error %s' % (data.shape, background_error.shape, analysis_error.shape))

    plt.plot(np.arange(length - 1), background_error, label='Forecast Error')
    plt.plot(np.arange(length - 1), analysis_error, label='Posterior Error')
    plt.xlabel('Time Steps (s)')
    plt.ylabel('MAE')
    plt.title('MAE Comparison, Forecast vs Posterior Error')
    plt.legend()
    plt.show()

def surface_figure():
    N = 40
    F = 5
    T = 100
    data = generate_l96(N, F, T, 0.01)

    x = np.arange(N)
    y = np.arange(T) * 0.01
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, data, cmap='viridis')
    ax.set_xlabel('Position')
    ax.set_ylabel('Time')
    ax.set_zlabel('Magnitude')
    plt.show()

if __name__ == '__main__':
    #snapshot_figure()
    surface_figure()
