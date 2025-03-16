import gymnasium as gym
import numpy as np
import data_generation

class ComponentRLEnv(gym.Env):
    def __init__(self, observations, updates, initial_ensemble, start_pos=0, score_type='rmse', action_space=None, observation_space=None):
        self.observations = observations
        self.updates = updates
        self.initial_ensemble = initial_ensemble
        self.start_pos = start_pos
        self.score_type = score_type

        self.action_space = action_space
        self.observation_space = observation_space

        self.identifier = 'ComponentRLEnv'

    def score(self, truth, action):
        if self.score_type == 'rmse':
            score = -np.sqrt(np.mean((truth - action)**2))
        elif self.score_type == 'supnorm':
            score = -np.linalg.norm(truth - action, ord=np.inf)
        elif self.score_type == 'logsupnorm':
            score = -np.log(np.linalg.norm(truth - action, ord=np.inf))
        elif self.score_type == 'manhattan':
            score = -np.linalg.norm(truth - action, ord=1)
        return score

    def step(self, action):
        truth = self.updates[self.pos]
        score = self.score(action, truth)
        self.pos += 1
        if self.pos < self.observations.shape[0]:
            observation = np.concat([self.observations[self.pos], truth])
            return observation, score, False, False, {}
        return np.concat([self.observations[0], truth]), score, True, False, {} # Terminate

    def reset(self, seed=False):
        self.pos = self.start_pos
        return np.concat([self.observations[self.pos], self.initial_ensemble]), {}


def generate_envs(l96_args, initial_condition, ensemble_condition, Nens, noise, inflation_coef=1.1, localization_coef=3, **kwargs):
    observations, ensembles, initial_ensembles = data_generation.generate_training_data(
        l96_args=l96_args,
        initial_condition=initial_condition,
        ensemble_condition=ensemble_condition,
        Nens=Nens,
        noise=noise,
        inflation_coef=inflation_coef,
        localization_coef=localization_coef
    )

    environment_functions = [
        lambda: ComponentRLEnv(observations, ensembles[i], initial_ensembles[i], **kwargs)
        for i in range(Nens)
    ]
    return environment_functions
