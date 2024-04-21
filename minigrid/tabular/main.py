import numpy as np
from agent import Q_learner, plotter


ENV_TYPE = "key"
LR = 0.2
DS = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.02
DECAY = "exp"
MAX_LEN = 250
STATE_TYPE=1
VERBOSE = True
LOG = 100



learner = Q_learner(ENV_TYPE,LR,DS,EPSILON_START,EPSILON_END,DECAY,1,MAX_LEN,STATE_TYPE,VERBOSE,LOG)


q,policy = learner.train(10000)
total_reward, steps =learner.test_agent()

print(np.mean(total_reward))      
print(np.mean(steps))      
plots = plotter(q,"empty")

plots.reward_step_plot(learner.rewards_per_episode,learner.steps_per_episode)
plots.reward_step_plot(total_reward,steps)
# plots.heatmap(learner.Q)