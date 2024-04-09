import time
import helpers
import matplotlib.pyplot as plt

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple

# from __future__ import annotations
from gym_minigrid.minigrid import COLOR_NAMES
from gym_minigrid.minigrid import Grid
from gym_minigrid.minigrid import MissionSpace
from gym_minigrid.minigrid import Door, Goal, Key, Wall, Lava, Floor
from minigrid_x import MiniGridEnv
from gym import spaces
import random
import cv2
from collections import deque, namedtuple
import numpy as np

from agent import DQN, DDQN
import warnings
warnings.filterwarnings("ignore", message=".*gym_minigrid.*") # suppress warnings of the minigrid environment

def preprocess_obs(obs):
    #get width and height of the image
    width, height = obs.shape[1], obs.shape[0]
    # crop image by 10% from all sides
    obs = obs[int(height/10):int(height-height/10), int(width/10):int(width-width/10)]
    #resize image to 96X96
    obs = cv2.resize(obs, (84, 84))
    #convert to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
    # add dimension
    return obs

env = helpers.EMPTYRGBImgObsWrapper(helpers.RandomEmptyEnv_10 (render_mode='rgb_array',max_steps=250))

obs = env.reset()[0]
screen = env.render()
obs = preprocess_obs(obs)
state_size = obs.shape


def dqn(agent, n_episodes=10000, max_t=1600, eps_start=1.0, eps_end=0.02, eps_decay=0.995):
    scores,ts = [],[]                  # list containing scores from each episode
    scores_window,ts_window = deque(maxlen=100), deque(maxlen=100) # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        initial_state, _ = env.reset()
        # state = np.reshape(initial_state, [1, state_size])
        state = preprocess_obs(initial_state)
        score = 0
        for t in range(max_t):
            action = agent.act(state)  # Choose an action
            next_state, reward, done, _, _ = env.step(action)  # Adjusted to match the five return values
            next_state = preprocess_obs(next_state)
            if(reward == 0.0):
                reward = -0.1
            # next_state = np.reshape(next_state, [1, state_size])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        ts_window.append(t)
        ts.append(t)
        if((agent.name == "DQN" and i_episode >5) or agent.name != "DQN"):
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('Episode {}\tAverage Score: {:.2f}\tAverage steps: {:.2f}\tEpsilon {:.2f} \n'.format(i_episode, np.mean(scores_window), np.mean(ts_window),eps), end="")
        if i_episode % 100 == 0:
            T.save(agent.network_local.state_dict(), f'checkpoint_{agent.name}{i_episode}.pth')
            
    return scores,ts


# #DDQN
# start_time = time.time()  # Start the timer
# agent = DDQN(state_size=obs.shape, action_size=3, seed=0)
# scores_ddqn,timesteps_ddqn = dqn(agent,n_episodes=2000)
# # # plot the scores
# fig, (ax1, ax2) = plt.subplots(2)
# fig.suptitle('Scores and Timesteps')
# ax1.plot(np.arange(len(scores_ddqn)), scores_ddqn)
# ax1.set_ylabel('Score')
# ax1.set_xlabel('Episode #')
# ax2.plot(np.arange(len(timesteps_ddqn)), timesteps_ddqn)
# ax2.set_ylabel('Timesteps')
# ax2.set_xlabel('Episode #')
# plt.savefig('ddqn_scores_timesteps.png')
# plt.show()

# end_time = time.time()  # Stop the timer
# elapsed_time = end_time - start_time  # Calculate elapsed time
# print(f"DDQN - Time taken: {elapsed_time} seconds")

start_time = time.time()  # Start the timer
agent = DQN(state_size=obs.shape, action_size=3, seed=0)
scores_dqn, timesteps_dqn = dqn(agent,n_episodes=2000)

end_time = time.time()  # Stop the timer
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"DQN - Time taken: {elapsed_time} seconds")


# two subplots one for scores and one for timesteps

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Scores and Timesteps')
ax1.plot(np.arange(len(scores_dqn)), scores_dqn)
ax1.set_ylabel('Score')
ax1.set_xlabel('Episode #')
ax2.plot(np.arange(len(timesteps_dqn)), timesteps_dqn)
ax2.set_ylabel('Timesteps')
ax2.set_xlabel('Episode #')
plt.savefig('dqn_scores_timesteps.png')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_dqn)), scores_dqn)
plt.plot(np.arange(len(timesteps_dqn)), timesteps_dqn)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('dqn_scores.png')



# Graph comparison of DQN and DDQN
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores_dqn)), scores_dqn, label='DQN')
# plt.plot(np.arange(len(scores_ddqn)), scores_ddqn, label='DDQN')
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.legend()
# plt.savefig('dqn_vs_ddqn.png')
