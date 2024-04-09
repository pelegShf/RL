import gym
from matplotlib import pyplot as plt
from agent import Agent
import numpy as np


# TODO: add plotting
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.01)
    scores, eps_history = [], []
    max_steps = 500
    n_games = 1000
    for i in range(n_games):
        done = False
        score = 0
        observation,_ = env.reset()
        for r in range(max_steps):
            action = agent.choose_action(observation)
            observation_, reward, done,_, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            if(done):
                break
        if(i % 10 == 0):
            agent.update_target_network()
      
        scores.append(score)
        eps_history.append(agent.epsilon)
        agent.epsilon = agent.epsilon - agent.eps_dec if agent.epsilon > agent.eps_end else agent.eps_end

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))

    # Plotting
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander.png'
    plt.plot(x, scores, eps_history)
    plt.xlabel('Episodes')
    plt.ylabel('Score')

    plt.savefig(filename)
    plt.show()
    print("Done")