import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from mlagents_envs.environment import UnityEnvironment

class ARForestDBHEnv(gym.Env):
    def __init__(self, unity_env):
        super(ARForestDBHEnv, self).__init__()
        self.unity_env = unity_env
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = self.reset()

    def reset(self):
        self.unity_env.reset()
        self.state = self.unity_env.get_state()
        return self.state

    def step(self, action):
        self.state, reward, done, info = self.unity_env.step(action)
        return self.state, reward, done, info

    def render(self, mode='human'):
        # This can be visually rendered in Unity
        pass

# Initialize Unity environment (you need to have Unity running with the AR Forest scene loaded)
unity_env = UnityEnvironment(file_name="ARForest")
env = ARForestDBHEnv(unity_env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

model.save("ppo_ar_forest_dbh")

# Test the trained model
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
