import gymnasium as gym
import numpy as np
import data_generation

class ComponentRLEnv(gym.Env):
    def __init__(self, observations, updates, initial_ensemble, start_pos=0, score_type='rmse', action_space=None, observation_space=None, rank=0):
        if isinstance(observations, np.ndarray):
            observations = [observations]
        if isinstance(updates, np.ndarray):
            updates = [updates]
        if isinstance(initial_ensemble, np.ndarray):
            initial_ensemble = [initial_ensemble]

        self.observations_backlog = observations
        self.updates_backlog = updates
        self.initial_ensemble_backlog = initial_ensemble
        self.start_pos = start_pos
        self.score_type = score_type

        self.observations = None
        self.updates = None
        self.initial_ensemble = None
        self.backlog_idx = None
        self.backlog_count = len(self.observations_backlog)

        self.action_space = action_space
        self.observation_space = observation_space

        self.identifier = 'ComponentRLEnv'
        self.rank_array = np.array([rank])

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
            observation = np.concat([self.rank_array, self.observations[self.pos], truth])
            return observation, score, False, False, {}
        return np.concat([self.rank_array, self.observations[0], truth]), score, True, False, {} # Terminate

    def reset(self, seed=False):
        self.pos = self.start_pos
        if self.backlog_idx == None:
            self.backlog_idx = 0
        else:
            self.backlog_idx = (self.backlog_idx + 1) % self.backlog_count
        self.observations = self.observations_backlog[self.backlog_idx]
        self.updates = self.updates_backlog[self.backlog_idx]
        self.initial_ensemble = self.initial_ensemble_backlog[self.backlog_idx]
        return np.concat([self.rank_array, self.observations[self.pos], self.initial_ensemble]), {}


def generate_envs(l96_args, initial_condition, ensemble_condition, Nens, noise, backlog_count=1, inflation_coef=1.1, localization_coef=3, **kwargs):
    observations_backlog = []
    updates_backlog = [[] for _ in range(Nens)]
    initial_ensembles_backlog = [[] for _ in range(Nens)]

    obs_max, obs_min, act_max, act_min = 0, 0, 0, 0 #observation and action space bounds
    for _ in range(backlog_count):
        observations, ensembles, initial_ensembles = data_generation.generate_training_data(
            l96_args=l96_args,
            initial_condition=initial_condition,
            ensemble_condition=ensemble_condition,
            Nens=Nens,
            noise=noise,
            inflation_coef=inflation_coef,
            localization_coef=localization_coef
        )
        observations_backlog.append(observations)
        obs_min = min(obs_min, np.min(observations))
        obs_max = max(obs_max, np.max(observations))
        for i in range(Nens):
            updates_backlog[i].append(ensembles[i])
            initial_ensembles_backlog[i].append(initial_ensembles[i])
            act_min = min(act_min, np.min(ensembles[i]))
            act_max = max(act_max, np.max(ensembles[i]))

    N = l96_args[0]
    act_center = (act_max + act_min) / 2
    act_min = act_center - abs(act_min - act_center) * 1.1 # give some leeway
    act_max = act_center + abs(act_max - act_center) * 1.1

    obs_center = (obs_max + obs_min) / 2
    obs_min = obs_center - abs(obs_min - obs_center) * 1.1
    obs_max = obs_center + abs(obs_max - obs_center) * 1.1
    # Need to specify observation ranges separately bc we pass rank and posterior ensembles to the model asw
    obs_low = np.concat([0, np.ones(N) * obs_min, np.ones(N) * act_min])
    obs_high = np.concat([Nens - 1, np.ones(N) * obs_max, np.ones(N) * act_max])

    action_space = gym.spaces.Box(low=act_min, high=act_max, shape=(N,), dtype=np.float32)
    observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    environment_functions = [
        lambda: ComponentRLEnv(observations, updates_backlog[i], initial_ensembles_backlog[i], action_space=action_space, observation_space=observation_space, rank=i, **kwargs)
        for i in range(Nens)
    ]
    return environment_functions
