from collections import deque, namedtuple
import random
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import cv2

from networks import network, mini_network

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size,device, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():
    def __init__(self, env,loss,seed,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000):
        self.env = env
        self.env_name = env.__class__.__name__
        self.state_size = self.__set_state_size()
        self.action_size = self.env.action_space.n
        self.action_range = np.arange(self.action_size)
        # reward shaping based on the environment
        self.reward_shaping = self.__simple_reward if self.env_name == "EMPTYRGBImgObsWrapper" else self.__reward_shape_key
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_update_rate = soft_update
        self.LR = LR
        self.update_every = update_every
        self.replay_buffer_size = replay_buffer_size
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.network_local = network().to(self.device)
        self.optimizer = optim.Adam(self.network_local.parameters(), lr = LR)
        self.loss = loss
        self.memory = ReplayBuffer(self.action_size, replay_buffer_size, self.batch_size,device=self.device,seed= seed)
        
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def act(self, state, eps=0.):
        if(len(state.shape) == 2):
            state = np.expand_dims(state, axis=0)
        state = T.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.network_local.eval()
        with T.no_grad():
            #Forward step
            action_values = self.network_local(state)
        self.network_local.train()

        # Epsilon-greedy action selection
        rand = random.random()
        if random.random() > eps:
            return T.argmax(action_values).item()
        else:
            action = random.choice(self.action_range)
            return action
   
    def train(self, n_episodes=10000, max_t=1600, eps_start=1.0, eps_end=0.02, eps_decay=0.995):
        scores,ts = [],[]                  # list containing scores from each episode
        scores_window,ts_window = deque(maxlen=100), deque(maxlen=100) # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            initial_state, _ = self.env.reset()
            state = self.preprocess_obs(initial_state)
            score = 0
            for t in range(max_t):
                action = self.act(state,eps=eps)  # Choose an action
                if(self.env_name != "EMPTYRGBImgObsWrapper"):
                    hasKey = int(self.env.is_carrying_key())
                    isDoorOpen = int(self.env.is_door_open())
                next_state, reward, done, _, _ = self.env.step(action)  # Adjusted to match the five return values
                next_state = self.preprocess_obs(next_state)
                
                if(self.env_name == "EMPTYRGBImgObsWrapper"):
                    reward = self.reward_shaping(reward)
                else:
                    hasKey_tag = int(self.env.is_carrying_key())
                    isDoorOpen_tag = int(self.env.is_door_open())
                    reward = self.reward_shaping(hasKey_tag , hasKey,isDoorOpen_tag , isDoorOpen)

                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            ts_window.append(t)
            ts.append(t)
            if((self.name == "DQN" and i_episode >25) or self.name != "DQN"):
                eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('Episode {}\tAverage Score: {:.2f}\tAverage steps: {:.2f}\tEpsilon {:.2f} \n'.format(i_episode, np.mean(scores_window), np.mean(ts_window),eps), end="")
            if i_episode % 100 == 0:
                T.save(self.network_local.state_dict(), f'checkpoint_{self.name}{i_episode}.pth')
                
        return scores,ts

    def __preprocess_obs(self,obs):
        width, height = obs.shape[1], obs.shape[0] #get width and height of the image
        obs = obs[int(height/10):int(height-height/10), int(width/10):int(width-width/10)]  # crop image by 10% from all sides
        obs = cv2.resize(obs, (84, 84))    #resize image
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)   #convert to grayscale
        obs = obs / 255.0 # Normlize state values to [0,1]
        return obs
    
    def __set_state_size(self):
        obs = self.env.reset()[0]
        obs = self.__preprocess_obs(obs)
        return obs.shape

    def __simple_reward(self,reward):
        if reward == 0.0:
            return -0.1
        return reward

    def __reward_shape_key(self,hasKey_tag , hasKey,isDoorOpen_tag , isDoorOpen):
        if(hasKey_tag > hasKey or isDoorOpen_tag > isDoorOpen):
            return 2
        elif(isDoorOpen_tag  < isDoorOpen or hasKey_tag < hasKey):
            return -1
        else:
            return -0.1
        
class DQN(Agent):
    def __init__(self,env,loss=F.mse_loss,seed=0,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000):
        super().__init__(env,loss,seed,batch_size,gamma,soft_update,LR,update_every,replay_buffer_size)
        self.name = "DQN"


    def learn(self, experiences, gamma):
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)
        q_targets_next = self.network_local(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        q_expected = self.network_local(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        # loss = F.mse_loss(q_expected, q_targets)
        # loss = F.huber_loss(q_expected, q_targets)
        loss = self.loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQN(Agent):
    def __init__(self,env,loss=F.mse_loss,seed=0,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000):
        super().__init__(env,loss,seed,batch_size,gamma,soft_update,LR,update_every,replay_buffer_size)
        
        self.name = "DDQN"
        self.network_target = mini_network().to(self.device)



    def learn(self, experiences, gamma):
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)
        q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)
        q_expected = self.network_local(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        loss = self.loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.network_local, self.network_target,self.soft_update_rate)

    def soft_update(self, local_model, target_model, SOFT_UPDATE_RATE):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(SOFT_UPDATE_RATE*local_param.data + (1.0-SOFT_UPDATE_RATE)*target_param.data)


class REINFORCE(Agent):
    def __init__(self,env,loss=F.mse_loss,seed=0,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000):
        super().__init__(env,loss,seed,batch_size,gamma,soft_update,LR,update_every,replay_buffer_size)
        self.name = "REINFORCE"
    
    
    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = T.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        
        self.policy_network.optimizer.zero_grad()
        policy_gradient = T.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.policy_network.optimizer.step()

    def train(self,n_episodes,max_t):
        numsteps = []
        avg_numsteps = []
        all_rewards = []
        scores,ts = [],[]                  # list containing scores from each episode
        scores_window,ts_window = deque(maxlen=100), deque(maxlen=100) # last 100 scores
            
        for episode in range(1,n_episodes+1):
            state,_ = self.env.reset()
            state = self.preprocess_obs(state)
            log_probs = []
            rewards = []
            score = 0
            for steps in range(max_t):
                state = np.expand_dims(state, axis=0)
                action, log_prob = self.policy_net.get_action(state)
                new_state, reward, done, _,_ = self.env.step(action)
                if(reward == 0.0):
                    reward = -0.1
                new_state = self.preprocess_obs(new_state)
                log_probs.append(log_prob)
                rewards.append(reward)
                score += reward


                if done:
                    self.update_policy( rewards, log_probs)
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
