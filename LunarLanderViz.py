import gymnasium as gym
from stable_baselines3 import DQN
import time

# Load trained model
model = DQN.load("./notebooks/dqn_lunar_stable8")

# Create an environment with rendering enabled
env = gym.make("LunarLander-v3", render_mode="human")

# Run a few episodes to see different scenarios
num_episodes = 5

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    
    print(f"\n=== Episode {episode + 1} ===")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.02)
        
        episode_reward += reward
        step_count += 1
        done = terminated or truncated
    
    print(f"Episode {episode + 1} finished after {step_count} steps")
    print(f"Total reward: {episode_reward:.2f}")

    time.sleep(1)

env.close()
print("\nVisualization complete!")