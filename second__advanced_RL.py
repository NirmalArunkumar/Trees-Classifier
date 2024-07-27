import gym
from gym import spaces
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_checker import check_env

class AdvancedDBHEnv(gym.Env):
    def __init__(self):
        super(AdvancedDBHEnv, self).__init__()
        # Continuous action space: movement (2D), altitude change, focus, zoom
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1]), high=np.array([1, 1, 1, 1, 1]), dtype=np.float32)
        # Observation space includes drone coordinates, camera settings, light, weather
        self.observation_space = spaces.Box(low=np.array([0, 0, 10, 0, 0, 0, 0]), high=np.array([100, 100, 200, 1, 1, 1, 1]), dtype=np.float32)
        self.state = self.reset()

    def reset(self):
        self.state = np.array([50, 50, 50, 0.5, 0.5, 0.5, 0.5])  # Middle of area, mid-altitude, medium camera settings, moderate weather
        return self.state

    def step(self, action):
        move_x, move_y, alt_change, focus, zoom = action
        x, y, alt, current_focus, current_zoom, light, weather = self.state
        x += move_x
        y += move_y
        alt += alt_change
        focus = np.clip(current_focus + focus, 0, 1)
        zoom = np.clip(current_zoom + zoom, 0, 1)

        # Simulate environmental changes
        light = np.random.uniform(0.4, 0.6)  # Random fluctuation
        weather = np.random.uniform(0.4, 0.6)  # Random fluctuation

        # Update state
        self.state = np.array([x, y, alt, focus, zoom, light, weather])
        # Dummy image quality calculation
        quality = focus * zoom * (1 - weather) * light
        # Reward is quality minus energy cost (simplified)
        reward = quality - 0.1 * (abs(move_x) + abs(move_y) + abs(alt_change))
        done = False

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")

# Check and train the environment
env = AdvancedDBHEnv()
check_env(env)

# Add action noise for exploration
action_noise = NormalActionNoise(mean=np.zeros(5), sigma=0.1 * np.ones(5))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=50000)

# Save the model
model.save("td3_advanced_dbh_model")

# Testing the trained model
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        break
