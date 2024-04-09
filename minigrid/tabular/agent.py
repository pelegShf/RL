from collections import deque, namedtuple
import random
import torch as T
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks import network

# TODO add this to the class agent as a parameter
device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
#BUFFER SIZE VERY IMPORTANT TO BE LARGE FOR CONVERGENCE
REPLAY_BUFFER_SIZE =    10000    #@param {type:"number"}
BATCH_SIZE         =    64       # minibatch size
GAMMA              =    0.99     # discount factor
SOFT_UPDATE_RATE   =    1e-3     # for soft update of target parameters
LEARNING_RATE      =    5e-4     # learning rate
UPDATE_EVERY       =    5        #@param {type:"slider", min:5, max:50}

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = T.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = T.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = T.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = T.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = T.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = np.arange(self.action_size)
        self.seed = random.seed(seed)
        self.network_local = network().to(device)

        self.optimizer = optim.Adam(self.network_local.parameters(), lr = LEARNING_RATE)
        self.memory = ReplayBuffer(action_size, REPLAY_BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    
    def step(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        if(len(state.shape) == 2):
            state = np.expand_dims(state, axis=0)
        state = T.from_numpy(state).float().unsqueeze(0).to(device)
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

class DQN(Agent):
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)
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
        print("Loss: ", loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DDQN(Agent):
    def __init__(self, state_size, action_size, seed):
        super().__init__(state_size, action_size, seed)

        self.name = "DDQN"
        self.network_target = network().to(device)



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

        self.soft_update(self.network_local, self.network_target,SOFT_UPDATE_RATE)

    def soft_update(self, local_model, target_model, SOFT_UPDATE_RATE):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(SOFT_UPDATE_RATE*local_param.data + (1.0-SOFT_UPDATE_RATE)*target_param.data)