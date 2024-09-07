import gymnasium as gym
import utils.envs
# from ..utils.envs import *

def test_environment():

    env = gym.make('trading-v0')
    #env.time_cost_bps = 0

    Episodes=3
    obs = []

    print(f"(sequence_length, n_feautres): {env.reset()[0].shape}")

    for _ in range(Episodes):
        observation = env.reset()[0]
        terminated, truncated = False, False
        count = 0

        while not terminated and not truncated:
            action = env.action_space.sample() # random
            observation, reward, terminated, truncated, info = env.step(action)
            obs = obs + [observation]
            
            if terminated or truncated:
                print(reward)
                print(count)

            count += 1
        
        print("Episode finished after {} timesteps".format(count))

if __name__ == "__main__":
    test_environment()
