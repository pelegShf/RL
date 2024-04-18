import time
import helpers
import matplotlib.pyplot as plt

import torch as T
import torch.nn.functional as F

import cv2
from collections import deque
import numpy as np

from agent import DQN, DDQN
import warnings
warnings.filterwarnings("ignore", message=".*gym_minigrid.*") # suppress warnings of the minigrid environment

def preprocess_obs(obs):
    width, height = obs.shape[1], obs.shape[0] #get width and height of the image
    obs = obs[int(height/10):int(height-height/10), int(width/10):int(width-width/10)]  # crop image by 10% from all sides
    obs = cv2.resize(obs, (84, 84))    #resize image
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)   #convert to grayscale
    obs = obs / 255.0 # Normlize state values to [0,1]
    return obs

env = helpers.EMPTYRGBImgObsWrapper(helpers.RandomEmptyEnv_10 (render_mode='rgb_array',max_steps=250))
# env = helpers.KEYRGBImgObsWrapper(helpers.RandomKeyMEnv_10 (render_mode='rgb_array',max_steps=250))

obs = env.reset()[0]
screen = env.render()
obs = preprocess_obs(obs)
state_size = obs.shape


start_time = time.time()  # Start the timer
agent = DQN(env,state_size=obs.shape,reward_shaping=helpers.reward_shape_key,loss=F.huber_loss, action_size=env.action_space.n,LR=0.0001, seed=0)

scores_dqn, timesteps_dqn = agent.train(n_episodes=5000, max_t=5000, eps_start=1.0, eps_end=0.02, eps_decay=0.995)

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
plt.savefig('ddqn_scores_timesteps.png')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_dqn)), scores_dqn)
plt.plot(np.arange(len(timesteps_dqn)), timesteps_dqn)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('ddqn_scores.png')

