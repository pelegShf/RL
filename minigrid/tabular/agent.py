import base64
import csv
from datetime import datetime
import IPython
from matplotlib import pyplot as plt
import numpy as np
import imageio
from helper import RandomEmptyEnv_10, RandomKeyMEnv_10, KeyFlatObsWrapper


class Q_learner:
  def __init__(self, env_type="empty",
               learning_rate=0.1,
               discount=0.99,
               epsilon_start=1.0,
               epsilon_end=0.05,
               epsilon_decay="linear",
               epsilon_mul=1.0,
               max_steps=250,
               relative=False,
               verbose=True,
               log_every=100,
               save_policy=True):

    self.env_type = env_type
    self.env = self.__set_env(env_type)
    self.action_space = self.env.action_space.n - 2 if env_type == "key" else self.env.action_space.n

    self.LR = learning_rate
    self.DS = discount
    self.eps = epsilon_start
    self.epsilon_start = epsilon_start
    self.epsilon_end = epsilon_end
    self.epsilon_mul = epsilon_mul # Only for mul decay
    self.decay = epsilon_decay
    self.verbose = verbose
    self.save_policy = save_policy



    self.HEIGHT = self.env.height-2
    self.WIDTH = self.env.width-2
    self.goal_options = 2 if env_type == "key" else 3
    self.DIRECTIONS = 4
    self.KEY_POS_HEIGHT = self.HEIGHT
    self.KEY_POS_WIDTH = 2 # Wall on third col always.
    self.DOOR_POS_HEIGHT = self.HEIGHT
    self.DOOR_OPEN = 2
    self.HAS_KEY = 2
    # self.Q = self.__create_Q()
    self.relative = relative
    self.Q = self.__create_Q()

    self.max_steps = max_steps
    self.log_every = log_every

    self.rewards_per_episode = []
    self.steps_per_episode = []
    self.dones = []
    self.is_holding_key = 0
    self.is_door_open = 0

    # Counters for debug
    self.opened = 0
    self.closed = 0
    self.finished = 0
    self.picked = 0

    if(self.verbose):
      print(f"Qshape: {self.Q.shape}")
      print(f"size: {np.prod(self.Q.shape)}")




  def episode(self,training=True):
    # Reset
    s0,_ = self.env.reset()
    done = False
    rewards = 0
    s = 0
    self.is_door_open = int(self.env.is_door_open())
    self.is_holding_key = int(self.env.is_carrying_key())

    state = self.__get_state()

    for s in range(self.max_steps):
      # Current state
      action = self.__behavior_policy(state)

      Q_sa = self.Q[state][action]
      action_for_env = self.__get_idx_action_env(action)
      obs, reward, done, _, _ = self.env.step(action_for_env)
      # Reward shaping
      reward= self.__reward_shaping(reward)
      # New state
      state_tag = self.__get_state()
      if(training): # Update Q
        action_tag = self.__target_policy(self.Q[state_tag])
        Q_sa_tag = self.Q[state_tag][action_tag]
        self.Q[state][action] = Q_sa + self.LR * (reward + self.DS*Q_sa_tag - Q_sa)

      #Update to new state
      state= state_tag

      rewards += reward
      if done:
        break

    self.rewards_per_episode.append(rewards)
    self.steps_per_episode.append(s)
    self.dones.append(done)


  def train(self,episodes):
    finished_episodes=0
    for e in range(1,episodes+1):
      if(e == 5000):
        self.LR = 0.1
      if(e == 10000):
        self.LR = 0.05
      self.episode()

                # Save rewards, steps, and done flag to a CSV file

      self.__decay_epsilon(episodes)

      if(self.verbose):
        finished_episodes = self.__log(e,episodes,finished_episodes)
    if(self.save_policy):
      self.save()
    # Save rewards, steps, and done flag to a CSV file after all episodes
    with open('episode_data.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Rewards", "Steps", "Done"])
      writer.writerows(zip(self.rewards_per_episode, self.steps_per_episode, self.dones))

     # update policy
    print("Training done")
    policy = np.argmax(self.Q, axis=1)
    print("Policy updated")
    return self.Q, policy


  def test_agent(self):
    # Reset the environment
   
    total_reward = []
    steps = []
    
    for e in range(100):
    
      r =0
      s=0
      state = self.env.reset()
      state = self.__get_state()
      done = False
      # Run the episode
      for s in range(self.max_steps):
          # Choose the action according to the policy
          action = np.argmax(self.Q[state])

          # Take the action
          next_state, reward, done, _ ,_= self.env.step(action)
          next_state = self.__get_state()
          reward = self.__reward_shaping(reward)
          # Update the total reward and state
          r += reward
          state = next_state

          # Increment the step count
          s += 1
          if(done):
            break
      total_reward.append(reward)
      steps.append(s)
      r=0
      s=0

    return total_reward, steps

  
  def create_video(self, policy, max_steps=1000, vid_name=''):
    with imageio.get_writer(vid_name, fps=10) as video:
      state = tuple(self.env.reset()[0])
      video.append_data(self.env.render())
      for s in range(max_steps):
        action = policy[state]
        next_state, r, done, _, _ = self.env.step(action)
        state = tuple(next_state)
        video.append_data(self.env.render())
        if done:
          break
    self.__embed_mp4(vid_name)


  def save(self):
    metadata = {
    'description': 'Q of env type key',
    'ENV_TYPE':self.env_type,
    'LR': self.LR,
    'DS': self.DS,
    'EPSILON_START': self.epsilon_start,
    'EPSILON_END': self.epsilon_end,
    'DECAY': self.decay,
    'MAX_STEP': self.max_steps,
    'shape': self.Q.shape,
    'relative': self.relative,
    }
    timestamp = datetime.now().strftime('%d%m%y_%H%M')
    np.savez(f'Q{self.env_type}_with_metadata_{timestamp}.npz', Q=self.Q, rewards=self.rewards_per_episode, steps=self.steps_per_episode, metadata=metadata)

  def load(self,data):
    try:
      if(isinstance(data, str)):
        data = np.load(f"data.npz",allow_pickle=True)
      self.Q = data['Q']
      self.rewards_per_episode = data['rewards']
      self.steps_per_episode = data['steps']

      metadata_arr = data['metadata']
      metadata_list = metadata_arr.tolist()
      metadata = dict(metadata_list)

      self.env_type = metadata['ENV_TYPE']
      self.env = self.__set_env(self.env_type)
      self.action_space = self.env.action_space.n - 2 if self.env_type == "key" else self.env.action_space.n
      self.LR = metadata['LR']
      self.DS = metadata['DS']
      self.eps = 0.05
      self.epsilon_start = metadata['EPSILON_START']
      self.epsilon_end = metadata['EPSILON_END']
      self.decay = metadata['DECAY']
      self.max_steps = metadata['MAX_STEP']
      self.relative = data['relative']

      self.HEIGHT = self.env.height-2
      self.WIDTH = self.env.width-2
      self.goal_options = 2 if self.env_type == "key" else 3

    except KeyError as e:
        print(f"KeyError: Missing key {e}")
        # Handle missing keys appropriately
    except Exception as e:
        print(f"An error occurred: {e}")
        # Handle other exceptions

  def __create_Q(self):
    if(self.relative):
      return self.__create_Q_relative()
    else:
      return self.__create_Q_abs()

  # Inner functions
  def __create_Q_abs(self):
    if(self.env_type == "key"):
      states = np.zeros((self.HEIGHT,self.WIDTH,self.DIRECTIONS,self.KEY_POS_HEIGHT,self.KEY_POS_WIDTH,self.DOOR_POS_HEIGHT,self.HAS_KEY,self.DOOR_OPEN))
    else:
      states = np.zeros((self.HEIGHT,self.WIDTH,self.DIRECTIONS,self.goal_options))
    Q = np.zeros_like(states)
    Q = np.expand_dims(Q,axis =-1)
    Q = np.repeat(Q, self.action_space, axis=-1)
    return Q
  
  def __create_Q_relative(self):
    if self.env_type == "key":
        # For "key" environment, include dimensions for key and door
        states = np.zeros((9, 8, 9, 8, 9, 8))
    else:
        # For other environments, only include dimensions for goal
        states = np.zeros((9, 8))
    Q = np.zeros_like(states)
    Q = np.expand_dims(Q,axis =-1)
    Q = np.repeat(Q, self.action_space, axis=-1)
    return Q
      

  def __get_state(self):
    if(self.relative):
      return self.__get_state_relative()
    else:
      return self.__get_state_abs()
    
  def __get_state_abs(self):
    x,y = self.env.get_position()
    heading = self.env.get_direction()
    goal_pos = self.__goal_to_idx(self.env.get_goal_pos()) # Maybe define it globl.

    state = (y-1,x-1,heading,goal_pos)
    if(self.env_type == "key"):
      key_posX,key_posY = self.env.get_k_pos()
      door_posX,door_posY = self.env.get_d_pos()
      state = ( y-1,x-1, heading, key_posY-1, key_posX-1, door_posY-1, int(self.is_holding_key),int(self.is_door_open))
    return state

  def __get_state_relative(self):
    x, y = self.env.get_position()
    heading = self.env.get_direction()

    # Calculate Manhattan distance and relative direction to the goal
    goal_x, goal_y = self.env.get_goal_pos()
    dist_goal, dir_goal = self.__calculate_distance_and_direction(x, y, goal_x, goal_y, heading)
    if dist_goal > 8:
        dist_goal = 8
    if self.env_type == "key":
        # Calculate Manhattan distance and relative direction to the key
        key_x, key_y = self.env.get_k_pos()
        dist_key, dir_key = self.__calculate_distance_and_direction(x, y, key_x, key_y, heading)
        if(dist_key > 8):
          dist_key = 8
        # Calculate Manhattan distance and relative direction to the door
        door_x, door_y = self.env.get_d_pos()
        dist_door, dir_door = self.__calculate_distance_and_direction(x, y, door_x, door_y, heading)
        if(dist_door > 8):
          dist_door = 8
        state = (dist_goal, dir_goal, dist_key, dir_key, dist_door, dir_door)
    else:
        state = (dist_goal, dir_goal)

    return state

  def __calculate_distance_and_direction(self, x, y, target_x, target_y, heading):
    # Calculate Manhattan distance
    distance = abs(x - target_x) + abs(y - target_y)

    # Calculate relative direction to the target
    dx, dy = target_x - x, target_y - y
    angle = np.arctan2(dy, dx) - heading  # Angle in radians
    angle = np.degrees(angle) % 360  # Convert to degrees and normalize to [0, 360)

    if 0 <= angle < 90:
      direction = 0  # straight ahead
    elif 90 <= angle < 180:
      direction = 1  # right
    elif 180 <= angle < 270:
      direction = 2  # back
    else:  # 270 <= angle < 360
      direction = 3  # left
    # Determine if the target is straight ahead, 45 to the right/left, etc.
    # if 0 <= angle < 45:
    #     direction = 0  # straight ahead
    # elif 45 <= angle < 90:
    #     direction = 1  # 45 to the right
    # elif 90 <= angle < 135:
    #     direction = 2  # 90 to the right
    # elif 135 <= angle < 180:
    #     direction = 3  # 135 to the right
    # elif 180 <= angle < 225:
    #     direction = 4  # back
    # elif 225 <= angle < 270:
    #     direction = 5  # 135 to the left
    # elif 270 <= angle < 315:
    #     direction = 6  # 90 to the left
    # else:  # 315 <= angle < 360
    #     direction = 7  # 45 to the left

    return distance, direction

  def __target_policy(self,actions):
    return np.argmax(actions)


  def __behavior_policy(self,state):
    actions = self.Q[state]
    if np.random.random() > self.eps:
      return np.argmax(actions)
    else:
        return np.random.randint(0,self.action_space)


  def __reward_shaping(self,reward):
    if(self.env_type == "key"):
      return self.__key_reward_shape(reward)
    else:
      return self.__empty_reward_shape(reward)


  def __empty_reward_shape(self,reward):
    if(reward == 0):
      return -0.1
    else:
      return  10


  def __key_reward_shape(self,reward):
    is_door_open_tag = self.env.is_door_open()
    is_holding_key_tag =  self.env.is_carrying_key()
    if(is_door_open_tag and not self.is_door_open):
      self.is_door_open = True
      self.opened += 1
      return 0.2
    elif (is_holding_key_tag and not self.is_holding_key): # closed door state
      self.is_holding_key = True
      self.picked +=1
      return  0.2
    elif (not is_door_open_tag and  self.is_door_open): # closed door state
      self.is_door_open = False
      self.closed +=1
      return -2
    elif(reward == 0): # normal state
      return -0.1
    else: # End state
      self.finished +=1
      return 10


  def __decay_epsilon(self,episodes):
    if self.decay == "linear":
      epsilon_delta = (self.epsilon_start - self.epsilon_end) / episodes
      self.eps = np.maximum(self.eps - epsilon_delta, self.epsilon_end)
    elif self.decay == "exp":
      decay_factor = (self.epsilon_end / self.epsilon_start) ** (1 / episodes)
      self.eps = np.maximum(self.eps * decay_factor, self.epsilon_end)
    elif self.decay == "mul":
      new_eps = self.eps * self.epsilon_mul
      self.eps = np.maximum(new_eps, self.epsilon_end)


  def __goal_to_idx(self,coord):
    if( coord[0]== 8 and coord[1] == 8):
      return 0
    elif ( coord[0]== 8 and coord[1] ==1):
      return 1
    else:
      return 2


  def __idx_to_goal(self,i):
    if( i == 0):
      return (8,8)
    elif ( i == 1):
      return (8,1)
    else:
      return (1,8)


  def __get_idx_action_env(self,action):
    if(action == 4):
      return 5
    if(action >4):
      print("Error")
    else:
      return action


  def __log(self,e,episodes,finished_episodes):
    if(self.steps_per_episode[-1] < self.max_steps-1):
      finished_episodes +=1
    if(e % self.log_every == 0 and e > 0):
      print(f"{e} / {episodes} Reward: {np.mean(np.array(self.rewards_per_episode)[-self.log_every:])} in {np.mean(np.array(self.steps_per_episode)[-self.log_every:])} steps. finished_episodes: {finished_episodes}/ {self.log_every}")
      print(f"Picked: {self.picked} Opened: {self.opened} Closed: {self.closed} Completed: {self.finished}")
      self.picked = 0
      self.opened = 0
      self.closed = 0
      self.finished = 0
      return 0
    return  finished_episodes


  def __set_env(self,env_type):
    if(env_type == "key"):
      env = KeyFlatObsWrapper(RandomKeyMEnv_10 (render_mode='rgb_array'))
    else:
      env = KeyFlatObsWrapper(RandomEmptyEnv_10 (render_mode='rgb_array'))
    return env

  def __embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
      <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)



class plotter:

  def __init__(self,Q,env_type):
    self.Q = Q
    self.env_type = env_type


  def sumQ(self,elem):
    if(self.env_type == "key"):
      if(elem == "direction"):
        Q_summed = np.sum(self.Q, axis=3) # KEY_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=3) # KEY_POS_WIDTH
        Q_summed = np.sum(Q_summed, axis=3) # DOOR_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=3) # HAS_KEY
        Q_summed = np.sum(Q_summed, axis=3) # DOOR_OPEN
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

      elif(elem == "door"):
        Q_summed = np.sum(self.Q, axis=2) # direction
        Q_summed = np.sum(Q_summed, axis=2) # KEY_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=2) # KEY_POS_WIDTH
        Q_summed = np.sum(Q_summed, axis=3) # HAS_KEY
        Q_summed = np.sum(Q_summed, axis=3) # DOOR_OPEN
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
      elif(elem == "key"):
        Q_summed = np.sum(self.Q, axis=2) # direction
        Q_summed = np.sum(Q_summed, axis=2) # KEY_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=2) # KEY_POS_WIDTH
        Q_summed = np.sum(Q_summed, axis=2) # DOOR_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=3) # DOOR_OPEN
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
      elif(elem == "door_open"):
        Q_summed = np.sum(self.Q, axis=2) # direction
        Q_summed = np.sum(Q_summed, axis=2) # KEY_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=2) # KEY_POS_WIDTH
        Q_summed = np.sum(Q_summed, axis=2) # DOOR_POS_HEIGHT
        Q_summed = np.sum(Q_summed, axis=2) # DOOR_OPEN
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

      Q_summed = np.sum(Q_summed, axis=3) # ACTIONS

    else:
        Q_summed = np.sum(self.Q, axis=2)
        Q_summed = np.sum(Q_summed, axis=3)  # Sum along the action dimension
        fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    return Q_summed,fig,axs




  def heatmap(self,elem):
    idx = -1

    Q_summed ,fig, axs = self.sumQ(elem)  # Sum along the action dimension
    # print(Q_summed.shape)
    axs = axs.flatten()
    for goal_state in range(Q_summed.shape[idx]):
        # ax = plt.subplot(gs[goal_state])
        ax = axs[goal_state]  # Get the current axis
        im = ax.imshow(Q_summed[:, :, goal_state], cmap='hot', interpolation='nearest', extent=[0, 8, 0, 8])  # Set extent parameter
        # ax.set_title(f'Action {self.__idx_to_goal(goal_state)}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')  # Set aspect ratio to 'equal' to ensure grid alignment
        ax.set_yticks(np.arange(8, -1, -1))  # Set Y-axis ticks from 8 to 0
        ax.set_yticklabels(np.arange(0, 9))  # Set Y-axis labels from 0 to 8

    # Hide the empty subplots
    for i in range(Q_summed.shape[-1], len(axs)):
        axs[i].axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label('Sum of Q-values')

    plt.show()

  def reward_step_plot(self,rewards_per_episode,steps_per_episode):
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot rewards per episode
    axs[0].plot(rewards_per_episode, label='Reward per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Reward per Episode')
    axs[0].legend()

    # Plot steps per episode
    axs[1].plot(steps_per_episode, label='Steps per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Number of Steps')
    axs[1].set_title('Steps per Episode')
    axs[1].legend()
    fig.tight_layout()
    plt.show()