from rl_env import RLEnv, RLEnsembleWiseEnv
from vectored_rl_env import ComponentRLEnv, generate_envs
from l96 import L96
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback
from construct_gc import construct_GC
import pickle

class Train:
    def __init__(self, N=40, F=5, Nens=20, action_coef=1, observation_coef=1, score='rmse', seed=0, debug=False, model=RLEnv, **kwargs):
        self.N = N
        self.F = F
        self.Nens = Nens

        self.l96_system = L96(self.N, self.F)
        derivative_func = lambda x0, t: self.l96_system.dx(x0, t) # pass self

        observation_dimension = self.N
        identity = np.identity(self.N, dtype=np.float32)
        noise = 0.1
        initial_condition = lambda : np.ones(self.N, dtype=np.float32) + np.random.multivariate_normal(np.zeros(self.N), 0.1 * identity)

        action_space = gym.spaces.Box(low=-1, high=1, shape=(self.Nens * self.N,), dtype="float32")
        observation_space = gym.spaces.Box(low=-10, high=10, shape=((self.Nens + 1) * self.N,), dtype="float32")

        #initial_ensemble_noise = (np.zeros(self.N), identity)
        initial_ensemble_noise = (np.zeros(self.N), np.eye(observation_dimension) * noise)
        termination_rule = lambda c, t, ens: c >= 500

        self.rl_environment = model(derivative_func, state_dimension=self.N, observation_dimension=self.N, Nens=self.Nens,
            action_space=action_space, observation_space=observation_space, H=identity, noise=noise, initial_condition=initial_condition,
            initial_ensemble_noise=initial_ensemble_noise, termination_rule=termination_rule, seed=seed, score=score, debug=debug, **kwargs)

# use for debugging
def value_callback(_locals, _globals):
    model = _locals['self']  # get ppo model
    obs = _locals['obs']  # current batch of observations
    values = model.policy.predict_values(obs)  # get critic estimates
    print(f"Mean predicted value: {values.mean():.2f}, Std: {values.std():.2f}")
    return True  # continue training

class CustomEvalCallback(BaseCallback):
    def __init__(self, env, timesteps, verbose=0, eval_freq=500):
        super().__init__(verbose)
        self.env = env
        self.eval_freq = eval_freq
        self.timesteps = timesteps

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        obs, info = self.env.reset()
        observations = []
        scores = []
        observations.append(obs)

        for i in range(self.timesteps):
            last_obs = observations[-1]
            prediction = self.model.predict(last_obs, deterministic=True)[0]
            next_obs, score, terminate, _, info = self.env.step(prediction)
            observations.append(next_obs)
            scores.append(score)
            if terminate:
                break

        scores = np.array(scores)
        obs_tensor = torch.tensor(np.array(observations), device='cpu')
        values = self.model.policy.predict_values(obs_tensor)
        if self.verbose: print(scores)
        print(f"Episode reward: {scores.sum():.2f}, Std: {scores.std():.2f} from {scores.shape[0]} timesteps")
        print(f"Mean predicted value: {values.mean():.2f}, Std: {values.std():.2f}")
        return True

def main():
    device = torch.device("cpu")

    def make_env(i):
        def wrapper():
            training_handler = Train(seed=i)
            training_handler
            return training_handler.rl_environment
        return wrapper

    n_epochs = 1500 # number of episodes
    epoch_length = 500 # length of each episode
    #eval_freq = 1500 # training steps before evaluating

    # wandb configs
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": epoch_length * n_epochs,
        "n_epochs": 1500,
        "N": 40,
        "Nens": 20,
        "F": 5,
        "score": "manhattan",
        "localization_matrix": construct_GC(3, 40, np.arange(40)),
        "inflation_coef": 3,
        "dt": 0.01,
        "ticks_per_step": 100
    }
    run = wandb.init(
        project="rl4da-1",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )

    training_env = Train(model=RLEnsembleWiseEnv, **config).rl_environment

    model = PPO("MlpPolicy", training_env,
        n_steps=epoch_length,
        n_epochs=n_epochs,
        batch_size=epoch_length,
        #batch_size=epoch_length // 10,
        gamma=0.99**(1/20),
        #gamma=0.98, # reduce time horizon
        ent_coef=0.01,
        #vf_coef=0.1,
        #clip_range_vf=0.2,
        #learning_rate=1e-5,
        clip_range=0.1,
        tensorboard_log=f"runs/{run.id}",
        verbose=2
    )
    model.learn(
        total_timesteps=config["total_timesteps"],
        log_interval=1,
        progress_bar=False,
        callback=WandbCallback(
            #gradient_save_freq=epoch_length,
            model_save_path=f"models/{run.id}",
            verbose=2
        )
    )
    model.save("lorenz96")

def vectored_main():
    device = torch.device("cpu")

    l96_args = (
        40, # N
        8,  # F
        500, # obs_freq
        0.01, # dt
        100 # timesteps
    )
    N = l96_args[0]
    Nens = 20
    noise = 0.1

    epoch_length = l96_args[2]
    n_epochs = 1500

    #action_space = gym.spaces.Box(low=-1, high=1, shape=(40,), dtype=np.float32)
    #observation_space = gym.spaces.Box(low=-1, high=1, shape=(80,), dtype=np.float32)
    initial_condition = lambda: np.ones(N) + np.random.multivariate_normal(np.zeros(N), noise * np.eye(N))
    ensemble_condition = lambda i: initial_condition()
    env_functions = generate_envs(l96_args, initial_condition, ensemble_condition, Nens, noise, backlog_count=10, inflation_coef=3)
    eval_env_functions = generate_envs(l96_args, initial_condition, ensemble_condition, Nens, noise, backlog_count=1, inflation_coef=3)

    env = SubprocVecEnv(env_functions)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = SubprocVecEnv(eval_env_functions)
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    config = {"policy_type": "MlpPolicy", "model_type": "ComponentRLModel", "total_timesteps": epoch_length * n_epochs,
        "n_epochs": n_epochs, "score": "rmse"}
    run = wandb.init(
        project="rl4da-1",
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True
    )
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=epoch_length,
        n_epochs=n_epochs,
        batch_size=epoch_length // 5,
        tensorboard_log=f"runs/{run.id}",
        verbose=2
    )

    # Callbacks
    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2
    )
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"models/{run.id}_eval_callback")
    callback = CallbackList([eval_callback, wandb_callback])

    model.learn(
        total_timesteps=epoch_length * n_epochs * Nens,
        log_interval=1,
        progress_bar=False,
        callback=callback
    )
    model.save(f"final_models/{run.id}")
    env.save(f"final_models/{run.id}-vec_normalize.pkl")

if __name__ == '__main__':
    #main()
    vectored_main()
