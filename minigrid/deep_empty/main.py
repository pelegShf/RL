import time
from networks import reinforce_network
import helpers
import matplotlib.pyplot as plt

import torch as T
import torch.nn.functional as F

import cv2
from collections import deque
import numpy as np

from agent import DQN, DDQN, REINFORCE
import warnings
warnings.filterwarnings("ignore", message=".*gym_minigrid.*") # suppress warnings of the minigrid environment


# env = helpers.EMPTYRGBImgObsWrapper(helpers.RandomEmptyEnv_10 (render_mode='rgb_array',max_steps=250))
env = helpers.KEYRGBImgObsWrapper(helpers.RandomKeyMEnv_10 (render_mode='rgb_array',max_steps=250))

#Init prints
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
print(device)
action_space = env.action_space
num_actions = action_space.n
print(f"Number of actions: {num_actions}")


# start_time = time.time()  # Start the timer
# # agent = DQN(env,loss=F.huber_loss,LR=0.0001, seed=0)
# agent = DDQN(env,loss=F.mse_loss,LR=0.0003,update_every=20,gamma=0.8, seed=0)
# # agent = REINFORCE(env=env,local_network=reinforce_network, seed=0)

# scores, timesteps = agent.train(n_episodes=5000, max_t=5000, eps_start=1.0, eps_end=0.02, eps_decay=0.995)
# # scores_reinforce, timesteps_reinforce = agent.train(n_episodes=5000, max_t=5000)

# end_time = time.time()  # Stop the timer
# elapsed_time = end_time - start_time  # Calculate elapsed time
# print(f"DQN - Time taken: {elapsed_time} seconds")


# two subplots one for scores and one for timesteps
def plot(scores,timesteps):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Scores and Timesteps')
    ax1.plot(np.arange(len(scores)), scores)
    ax1.set_ylabel('Score')
    ax1.set_xlabel('Episode #')
    ax2.plot(np.arange(len(timesteps)), timesteps)
    ax2.set_ylabel('Timesteps')
    ax2.set_xlabel('Episode #')
    plt.savefig('ddqn_scores_timesteps.png')
    plt.show()


def run_reinforce():


    start_time = time.time()  # Start the timer
    # agent = DQN(env,loss=F.huber_loss,LR=0.0001, seed=0)
    # agent = REINFORCE(env,loss=F.mse_loss,LR=0.0003,update_every=20,gamma=0.8, seed=0)
    agent = REINFORCE(env=env,local_network=reinforce_network, seed=0)

    scores, timesteps,dones = agent.train(n_episodes=2, max_t=5000)
    # scores_reinforce, timesteps_reinforce = agent.train(n_episodes=5000, max_t=5000)
    print(f"Done: {sum(dones)}/2 Avg. reward: {sum(scores)/len(scores)} Avg. length: {sum(timesteps)/len(timesteps)} ")
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"REINFORCE - Time taken: {elapsed_time} seconds")
    plot(scores,timesteps)

run_reinforce()