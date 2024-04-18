from collections import deque
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import helpers

from networks import  reinforce_network



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
    obs = obs / 255.0
    return obs


GAMMA = 0.9


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = T.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = T.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()

# env = helpers.EMPTYRGBImgObsWrapper(helpers.RandomEmptyEnv_10 (render_mode='rgb_array',max_steps=250))
env = helpers.EMPTYRGBImgObsWrapper(helpers.RandomEmptyEnv_10(render_mode='rgb_array',max_steps=250))

obs = env.reset()[0]
screen = env.render()
obs = preprocess_obs(obs)
state_size = obs.shape


policy_net = reinforce_network()

max_episode_num = 2000
max_steps = 2000
numsteps = []
avg_numsteps = []
all_rewards = []
scores,ts = [],[]                  # list containing scores from each episode
scores_window,ts_window = deque(maxlen=100), deque(maxlen=100) # last 100 scores
        
for episode in range(max_episode_num):
    state,_ = env.reset()
    state = preprocess_obs(state)
    log_probs = []
    rewards = []
    score = 0
    for steps in range(max_steps):
        state = np.expand_dims(state, axis=0)
        action, log_prob = policy_net.get_action(state)
        new_state, reward, done, _,_ = env.step(action)
        if(reward == 0.0):
            reward = -0.1
        new_state = preprocess_obs(new_state)
        log_probs.append(log_prob)
        rewards.append(reward)
        score += reward


        if done:
            update_policy(policy_net, rewards, log_probs)
            numsteps.append(steps)
            avg_numsteps.append(np.mean(numsteps[-10:]))
            break
        all_rewards.append(np.sum(rewards))

        state = new_state
    scores_window.append(score)       # save most recent score
    scores.append(score)              # save most recent score
    ts_window.append(steps)
    ts.append(steps)
    # print("episode: {}, total reward: {}, average_reward: {}, length: {}".format(episode, np.round(np.mean(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))

    print('Episode {}\tAverage Score: {:.2f}\tAverage steps: {:.2f} \n'.format(episode, np.mean(scores_window), np.mean(ts_window)), end="")

# plt.plot(numsteps)
# plt.plot(avg_numsteps)
# plt.xlabel('Episode')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(ts)), ts)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('ddqn_scores.png')

