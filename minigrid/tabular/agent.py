from collections import deque, namedtuple
import random
import torch as T
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import cv2

from networks import network, mini_network

# # TODO add this to the class agent as a parameter
# device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
# #BUFFER SIZE VERY IMPORTANT TO BE LARGE FOR CONVERGENCE
# REPLAY_BUFFER_SIZE =    10000    #@param {type:"number"}
# BATCH_SIZE         =    64       # minibatch size
# GAMMA              =    0.99     # discount factor
# SOFT_UPDATE_RATE   =    1e-3     # for soft update of target parameters
# LEARNING_RATE      =    5e-4     # learning rate
# UPDATE_EVERY       =    5        #@param {type:"slider", min:5, max:50}

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
    def __init__(self, env,state_size, action_size,seed,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = np.arange(self.action_size)
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
        self.memory = ReplayBuffer(action_size, replay_buffer_size, self.batch_size,device=self.device,seed= seed)

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
        if random.random() > eps:
            return T.argmax(action_values).item()
        else:
            return random.choice(self.action_range)
        
    def preprocess_obs(self,obs):
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
    
    def train(self, n_episodes=10000, max_t=1600, eps_start=1.0, eps_end=0.02, eps_decay=0.995):
        scores,ts = [],[]                  # list containing scores from each episode
        scores_window,ts_window = deque(maxlen=100), deque(maxlen=100) # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            initial_state, _ = self.env.reset()
            # state = np.reshape(initial_state, [1, state_size])
            state = self.preprocess_obs(initial_state)
            score = 0
            for t in range(max_t):
                action = self.act(state)  # Choose an action
                next_state, reward, done, _, _ = self.env.step(action)  # Adjusted to match the five return values
                next_state = self.preprocess_obs(next_state)
                if(reward == 0.0):
                    reward = -0.1
                # next_state = np.reshape(next_state, [1, state_size])
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            ts_window.append(t)
            ts.append(t)
            if((self.name == "DQN" and i_episode >5) or self.name != "DQN"):
                eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('Episode {}\tAverage Score: {:.2f}\tAverage steps: {:.2f}\tEpsilon {:.2f} \n'.format(i_episode, np.mean(scores_window), np.mean(ts_window),eps), end="")
            if i_episode % 100 == 0:
                T.save(self.network_local.state_dict(), f'checkpoint_{self.name}{i_episode}.pth')
                
        return scores,ts

class DQN(Agent):
    def __init__(self,env,state_size, action_size,seed,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000):
        super().__init__(env,state_size, action_size,seed,batch_size=64,gamma=0.99,soft_update=1e-3,LR=5e-4,update_every=5,replay_buffer_size=10000)
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
        loss = F.huber_loss(q_expected, q_targets)
        # print("Loss: ", loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQN(Agent):
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

        self.name = "DDQN"
        self.network_target = mini_network().to(self.device)



    def learn(self, experiences, gamma):
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)
        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        # detach(): This method is used to detach the output from the computation graph.
        # In PyTorch, when you perform operations on tensors, a computation graph is created for backpropagation.
        # However, for the target Q-values, you don't need gradients, as these are used as fixed targets during the training of the local network.
        #Detaching them from the graph means that operations on these tensors won't track gradients, which can save memory and computation.
        q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)

        ### Calculate target value from bellman equation
        q_targets = rewards + gamma * q_targets_next * (1 - dones)

        ### Calculate expected value from local network
        #Suppose self.qnetwork_local(states) returns a tensor of shape [batch_size, num_actions],
        #representing the Q-values for each action in each state for a batch of state observations.
        #If actions is a tensor of shape [batch_size, 1],
        # where each element is the index of the action taken in the corresponding state,
        # then gather(1, actions) will return a tensor of shape [batch_size, 1],
        #where each element is the Q-value of the action taken in the corresponding state.
        # Forward on the local network
        q_expected = self.network_local(states).gather(1, actions)

        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.network_local, self.network_target,self.soft_update_rate)

    def soft_update(self, local_model, target_model, SOFT_UPDATE_RATE):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(SOFT_UPDATE_RATE*local_param.data + (1.0-SOFT_UPDATE_RATE)*target_param.data)