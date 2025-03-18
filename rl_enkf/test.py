from data_generation import *
import gymnasium as gym
import torch
from stable_baselines3 import PPO

def load_model(path):
    return PPO.load(path)


def compare_model_eakf(model, l96_args=(40, 8, 100, 0.01, 100), initial_condition=None, ensemble_condition=None, Nens=20, noise=0.1, inflation_coef=1.1, distance=rmse):
    N, F, obs_freq, dt, timesteps = l96_args
    system = L96(N, F)

    H = np.eye(N)
    R = np.eye(N) * noise
    ground_truth = initial_condition() # (N,) array of initial points
    ensembles = [ensemble_condition(i) for i in range(Nens)] # (N, Nens) array, this is the ensemble around each truth
    l96_data = generate_l96(N, F, timesteps * obs_freq + 1, dt) # generate trailing, as we don't use the first data point
    t = 0

    CMat = construct_GC(3, N, np.arange(0, N))

    background_error, analysis_error, model_error = [], [], []
    for i in tqdm(range(1, obs_freq)):
        priors = ensembles[:]
        for _ in range(timesteps):
            priors = [runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in priors]
            t += dt
        prior_mean = sum(priors) / N # prior mean
        gt = l96_data[i * timesteps, :] # ground truth
        background_error.append(distance(gt, prior_mean)) # forecast vs ground truth

        obs = H @ gt + np.random.multivariate_normal(np.zeros(N), R) # get observation
        prior_stack = np.stack(priors, 1)
        prior_means_stack = np.stack([prior_mean] * Nens, 1)
        inflated_priors = prior_means_stack + np.sqrt(inflation_coef) * (prior_stack - prior_means_stack) # inflate priorso

        true_posteriors = eakf(Nens, N, inflated_priors, H, noise, 1, CMat, obs) # get posterior distribution
        predicted_posteriors = []
        for i, prior in enumerate(priors):
            model_obs = np.concat([np.array([i]), obs, prior])
            post, _ = model.predict(model_obs)
            predicted_posteriors.append(post)
        posterior_mean = np.mean(true_posteriors, axis=1)
        model_posterior_mean = sum(predicted_posteriors) / Nens
        analysis_error.append(distance(gt, posterior_mean))
        model_error.append(distance(gt, model_posterior_mean))

        ensembles = np.unstack(true_posteriors, axis=1)
        #ensembles = np.moveaxis(new_posteriors, 1, 0) # my version of numpy is outdated

    return l96_data[0::timesteps][1:], background_error, analysis_error, model_error

def plot_diff():
    import matplotlib.pyplot as plt
    N = 40
    model = load_model("models/g59ronax_eval_callback/best_model.zip")
    initial_condition = lambda: np.ones(N) + np.random.multivariate_normal(np.zeros(N), 0.1 * np.eye(N))
    ensemble_condition = lambda i: initial_condition()
    data, background_error, analysis_error, model_error = compare_model_eakf(model, l96_args=(40, 8, 500, 0.01, 100), initial_condition=initial_condition, ensemble_condition=ensemble_condition)

    length = len(data)
    background_error = np.array(background_error)
    analysis_error = np.array(analysis_error)
    model_error = np.array(model_error)

    plt.plot(np.arange(length - 1), background_error, label="Forecast error")
    plt.plot(np.arange(length - 1), analysis_error, label="EnKF error")
    plt.plot(np.arange(length - 1), model_error, label="Model error")
    plt.legend()
    plt.show()

def plot_stepwise():
    import matplotlib.pyplot as plt
    model = load_model("models/g59ronax_eval_callback/best_model.zip")
    N, F, obs_freq, dt, timesteps = 40, 8, 500, 0.01, 100
    initial_condition = lambda: np.ones(N) + np.random.multivariate_normal(np.zeros(N), 0.1 * np.eye(N))
    ensemble_condition = lambda i: initial_condition()
    Nens = 20
    noise = 0.1
    inflation_coef = 1.1

    system = L96(N, F)

    H = np.eye(N)
    R = np.eye(N) * noise
    ground_truth = initial_condition() # (N,) array of initial points
    ensembles = [ensemble_condition(i) for i in range(Nens)] # (N, Nens) array, this is the ensemble around each truth
    l96_data = generate_l96(N, F, timesteps * obs_freq + 1, dt) # generate trailing, as we don't use the first data point
    t = 0

    arange = np.arange(N)

    CMat = construct_GC(3, N, np.arange(0, N))

    for i in tqdm(range(1, obs_freq)):
        priors = ensembles[:]
        for _ in range(timesteps):
            priors = [runge_kutta_4(system.dx, ensemble, t, dt) for ensemble in priors]
            t += dt
        prior_mean = sum(priors) / N # prior mean
        gt = l96_data[i * timesteps, :] # ground truth

        obs = H @ gt + np.random.multivariate_normal(np.zeros(N), R) # get observation
        prior_stack = np.stack(priors, 1)
        prior_means_stack = np.stack([prior_mean] * Nens, 1)
        inflated_priors = prior_means_stack + np.sqrt(inflation_coef) * (prior_stack - prior_means_stack) # inflate priorso

        true_posteriors = eakf(Nens, N, inflated_priors, H, noise, 1, CMat, obs) # get posterior distribution
        predicted_posteriors = []
        for i, prior in enumerate(priors):
            model_obs = np.concat([np.array([i]), obs, prior])
            post, _ = model.predict(model_obs)
            predicted_posteriors.append(post)
        posterior_mean = np.mean(true_posteriors, axis=1)
        model_posterior_mean = sum(predicted_posteriors) / Nens

        # plot
        plt.plot(arange, gt, label="ground truth")
        plt.plot(arange, obs, label="observation")
        plt.plot(arange, prior_mean, label="prior")
        plt.plot(arange, posterior_mean, label="true posterior")
        plt.plot(arange, model_posterior_mean, label="model posterior")
        plt.legend()
        plt.show()

        ensembles = np.unstack(true_posteriors, axis=1)
        #ensembles = np.moveaxis(new_posteriors, 1, 0) # my version of numpy is outdated

if __name__ == '__main__':
    torch.device('cpu')
    #plot_diff()
    plot_stepwise()
