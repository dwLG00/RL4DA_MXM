import gymnasium as gym
import numpy as np
from enkf import eakf

class RLEnv(gym.Env):
    def __init__(self, derivative_func, dt=0.1, state_dimension=None, observation_dimension=None, Nens=40, action_space=None, observation_space=None, H=None, noise=None, initial_condition=None,
        initial_ensemble_noise=(None, None), termination_rule=None, ground_truth_forward=None, seed=None, score=None, debug=None, **kwargs):
        super(RLEnv, self).__init__()

        self.debug = debug

        self.dx = derivative_func # dx(x, t) is derivative at time t at pos x
        self.dt = dt # each timestep
        self.state_dimension = state_dimension
        self.observation_dimension = observation_dimension
        self.ensemble_size = Nens # number of elements in the ensemble
        self.observation_matrix = H # projects each (ground truth) state vector to observation space
        #self.noise_covariance = noise # covariance matrix for observations. obs = H * truth + w, w ~ N(noise)
        self.noise_variance = noise
        self.noise_covariance = np.eye(observation_dimension) * noise
        self.initial_condition = initial_condition # function that returns a (possibly random?) initial condition
        self.initial_ensemble_noise = initial_ensemble_noise # (mean, covariance) of noise added to initial condition to generate initial ensemble members
        self.termination_rule = termination_rule # termination_rule(t, ensemble) returns whether or not to terminate per step
        self.ground_truth_forward = ( # Used to get the next ground truth state (when using a dataset). Otherwise just uses runge kutta 4.
            ground_truth_forward if ground_truth_forward != None
            else lambda x0, t, dt: runge_kutta_4(self.dx, x0, t, dt)
        )
        self.action_space = action_space
        self.observation_space = observation_space
        self.score_type = score if score != None else 'rmse'

        self.T = 0
        self.count = 0
        self.last_obs = None
        self.seed = seed

    def score(self, zens, action):
        if self.score_type == 'rmse':
            score = -np.sqrt(np.mean((zens - action)**2))
        elif self.score_type == 'supnorm':
            score = -np.linalg.norm(zens - action, ord=np.inf)
        elif self.score_type == 'logsupnorm':
            score = -np.log(np.linalg.norm(zens - action, ord=np.inf))
        elif self.score_type == 'manhattan':
            score = -np.linalg.norm(zens - action, ord=1)
        return score

    def step(self, action):
        H = self.observation_matrix

        # see how the enkf model would have updated the ensemble
        zens = np.stack(self.ensembles, 1)
        updated_zens = eakf(self.ensemble_size, self.observation_dimension, zens, H, self.noise_variance, False, None, self.last_obs)
        zens_diff = np.concatenate(np.unstack(zens - updated_zens, axis=1))

        ensemble_diffs = np.split(action, self.ensemble_size)
        new_ensemble = list(map(lambda t: t[0] + t[1], zip(self.ensembles, ensemble_diffs))) # add action to each ensemble member. Assumption is that the update per step is sufficiently small

        # get rmse between enkf-predicted update and action to get score
        zens = np.concatenate(np.unstack(updated_zens, axis=1))
        score = self.score(zens_diff, action)
        if self.count == 0 and self.debug:
            print('zens_diff: %s' % zens_diff)
            print('action: %s' % action)
            print('rmse score: %s' % score)

        # Now forward-pass the ground truth, compute forecast, and get observation
        self.ground_truth = self.ground_truth_forward(self.ground_truth, self.T, self.dt)

        # Compute forecast, observe next step to get error
        #self.ensembles = np.split(action, self.ensemble_size)
        self.ensembles = np.unstack(updated_zens, axis=1)
        priors = [
            runge_kutta_4(self.dx, ensemble, self.T, self.dt)
            for ensemble in self.ensembles
        ]
        forecast_mean = sum(priors) / self.ensemble_size

        observation = H @ self.ground_truth + self.np_random.multivariate_normal(np.zeros(self.observation_dimension), self.noise_covariance)
        error = H @ forecast_mean - observation # error between observation and prior
        self.last_obs = observation

        # Construct input vector (ensembles + error between forecast and observation)
        input_vector = np.concat((error, np.concat(self.ensembles)))
        #input_vector = np.concat((error, np.concat(priors)))
        self.T += self.dt
        self.count += 1


        return input_vector, score, self.termination_rule(self.count, self.T, self.ensembles), False, self.__get_info()

    def reset(self, seed=None, **kwargs):
        if not self.seed:
            super().reset(seed=seed)
        else:
            super().reset(seed=self.seed)

        self.T = 0

        ensemble_mean, ensemble_covariance = self.initial_ensemble_noise
        H = self.observation_matrix

        # initial condition
        self.ground_truth = self.initial_condition()
        self.ensembles = [self.ground_truth + self.np_random.multivariate_normal(ensemble_mean, ensemble_covariance) for _ in range(self.ensemble_size)]

        # compute forecast, observe next step to get error
        priors = [
            runge_kutta_4(self.dx, ensemble, self.T, self.dt)
            for ensemble in self.ensembles
        ]
        forecast_mean = sum(priors) / self.ensemble_size
        self.ground_truth = self.ground_truth_forward(self.ground_truth, self.T, self.dt)
        observation = H @ self.ground_truth + self.np_random.multivariate_normal(np.zeros(self.observation_dimension), self.noise_covariance)
        error = H @ forecast_mean - observation

        # construct our RL model input vector
        input_vector = np.concat((error, np.concat(self.ensembles)))
        self.last_obs = observation
        self.T += self.dt

        return input_vector, self.__get_info()

    def __get_info(self):
        return {
            "time": self.T,
            "ground_truth": self.ground_truth,
            "ens_mean": np.mean(self.ensembles)
        }


class RLEnsembleWiseEnv(gym.Env):
    '''
        Ensemble-wise RL Environment
        Instead of computing a single timestep with each step, each step will compute a single timestep _for a single ensemble member_.
        Only after computing all ensemble members will a reward be provided.
    '''
    def __init__(self, derivative_func, dt=0.1, state_dimension=None, observation_dimension=None, Nens=40, action_space=None, observation_space=None, observation_bounds=1, H=None, noise=None,
        initial_condition=None, initial_ensemble_noise=(None, None), termination_rule=None, ground_truth_forward=None, seed=None, score=None, debug=None, **kwargs):
        super(RLEnsembleWiseEnv, self).__init__()

        self.debug = debug

        self.dx = derivative_func # dx(x, t) is derivative at time t at pos x
        self.dt = dt # each timestep
        self.state_dimension = state_dimension
        self.observation_dimension = observation_dimension
        self.ensemble_size = Nens # number of elements in the ensemble
        self.observation_matrix = H # projects each (ground truth) state vector to observation space
        #self.noise_covariance = noise # covariance matrix for observations. obs = H * truth + w, w ~ N(noise)
        self.noise_variance = noise
        self.noise_covariance = np.eye(observation_dimension) * noise
        self.initial_condition = initial_condition # function that returns a (possibly random?) initial condition
        self.initial_ensemble_noise = initial_ensemble_noise # (mean, covariance) of noise added to initial condition to generate initial ensemble members
        self.termination_rule = termination_rule # termination_rule(t, ensemble) returns whether or not to terminate per step
        self.ground_truth_forward = ( # Used to get the next ground truth state (when using a dataset). Otherwise just uses runge kutta 4.
            ground_truth_forward if ground_truth_forward != None
            else lambda x0, t, dt: runge_kutta_4(self.dx, x0, t, dt)
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.state_dimension,), dtype=np.float32)
        #self.observation_space = gym.spaces.Tuple((
        #    gym.spaces.Discrete(self.ensemble_size),
        #    gym.spaces.Box(low=-observation_bounds, high=observation_bounds, shape=(self.state_dimension,), dtype=np.float32),
        #    gym.spaces.Box(low=-observation_bounds, high=observation_bounds, shape=(self.state_dimension,), dtype=np.float32)
        #))
        self.observation_space = gym.spaces.Box(
            low=np.array([0] + [-observation_bounds] * 2 * self.state_dimension),
            high=np.array([self.ensemble_size] + [observation_bounds] * 2 * self.state_dimension),
            dtype=np.float32
        )
        self.score_type = score if score != None else 'rmse'

        self.T = 0
        self.count = 0
        self.last_obs = None
        self.last_error = None
        self.true_ensemble_diff = []
        self.actions = []
        self.seed = seed

    def score(self, zens, action):
        if self.score_type == 'rmse':
            score = -np.sqrt(np.mean((zens - action)**2))
        elif self.score_type == 'supnorm':
            score = -np.linalg.norm(zens - action, ord=np.inf)
        elif self.score_type == 'logsupnorm':
            score = -np.log(np.linalg.norm(zens - action, ord=np.inf))
        elif self.score_type == 'manhattan':
            score = -np.linalg.norm(zens - action, ord=1)
        return score

    def step(self, action):
        index = (self.count - 1) % self.ensemble_size # 0-indexed ensemble index
        next_index = self.count % self.ensemble_size
        #true_ensemble_diff = self.true_ensemble_diff[index] # the actual EnKF updated ensemble value
        #self.errors.append(action - true_ensemble_diff)
        self.actions.append(action)

        if self.count % self.ensemble_size == 0: # we've processed each ensemble
        #if len(self.actions) == self.ensemble_size:
            H = self.observation_matrix
            # Compute score
            assert len(self.actions) == self.ensemble_size
            all_actions = np.concat(self.actions)
            all_diffs = np.concat(self.true_ensemble_diff)
            score = self.score(all_diffs, all_actions)

            # forward timestep pass
            self.ensembles = [sum(z) for z in zip(self.ensembles, self.true_ensemble_diff)] # remember, true_ensemble_diff is the difference between zens (= self.ensembles) and the true (kalman) ensemble
            priors = [
                runge_kutta_4(self.dx, ensemble, self.T, self.dt)
                for ensemble in self.ensembles
            ]
            forecast_mean = sum(priors) / self.ensemble_size
            self.ground_truth = self.ground_truth_forward(self.ground_truth, self.T, self.dt) # update the ground truth
            observation = H @ self.ground_truth + self.np_random.multivariate_normal(np.zeros(self.observation_dimension), self.noise_covariance) # get observation, errors, and set them
            error = H @ forecast_mean - observation
            self.last_error = error
            self.last_obs = observation

            # compute true ensemble update step
            zens = np.stack(self.ensembles, 1)
            true_ensemble = eakf(self.ensemble_size, self.observation_dimension, zens, H, self.noise_variance, False, None, self.last_obs)
            true_ensemble_diff = true_ensemble - zens
            self.true_ensemble_diff = np.unstack(true_ensemble_diff, axis=1)
            self.T += self.dt

            # reset things
            self.actions = []
        else:
            score = 0

        next_obs = np.concat([np.array([next_index]), self.last_error, self.ensembles[next_index]])
        #next_obs = (next_index, self.last_error, self.ensembles[next_index])
        self.count += 1

        return next_obs, score, self.termination_rule(self.count, self.T, self.ensembles), False, self.__get_info()

    def reset(self, seed=None, **kwargs):
        if not self.seed:
            super().reset(seed=seed)
        else:
            super().reset(seed=self.seed)

        self.T = 0
        self.count = 0
        self.last_obs = None
        self.last_error = None
        self.true_ensemble_diff = []
        self.actions = []

        ensemble_mean, ensemble_covariance = self.initial_ensemble_noise
        H = self.observation_matrix

        # initial condition
        self.ground_truth = self.initial_condition()
        self.ensembles = [self.ground_truth + self.np_random.multivariate_normal(ensemble_mean, ensemble_covariance) for _ in range(self.ensemble_size)]

        # compute forecast, observe next step to get error
        priors = [
            runge_kutta_4(self.dx, ensemble, self.T, self.dt)
            for ensemble in self.ensembles
        ]
        forecast_mean = sum(priors) / self.ensemble_size
        self.ground_truth = self.ground_truth_forward(self.ground_truth, self.T, self.dt)
        observation = H @ self.ground_truth + self.np_random.multivariate_normal(np.zeros(self.observation_dimension), self.noise_covariance)
        error = H @ forecast_mean - observation
        self.last_error = error

        self.last_obs = observation
        zens = np.stack(self.ensembles, 1)
        true_ensemble = eakf(self.ensemble_size, self.observation_dimension, zens, H, self.noise_variance, False, None, self.last_obs)
        true_ensemble_diff = true_ensemble - zens
        self.true_ensemble_diff = np.unstack(true_ensemble_diff, axis=1)

        # construct our RL model input vector
        #observation = (0, error, self.ensembles[0])
        observation = np.concatenate((np.array([0]), error, self.ensembles[0]))
        self.T += self.dt
        self.count += 1

        return observation, self.__get_info()

    def __get_info(self):
        return {
            "time": self.T,
            "ground_truth": self.ground_truth,
            "ens_mean": np.mean(self.ensembles)
        }

def runge_kutta_4(func, x0, t, dt):
    '''Apply runge-kutta to a function'''
    k1 = func(x0, t)
    k2 = func(x0 + (dt / 2.0) * k1, t + (dt / 2.0))
    k3 = func(x0 + (dt / 2.0) * k2, t + (dt / 2.0))
    k4 = func(x0 + dt * k3, t + dt)

    return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

