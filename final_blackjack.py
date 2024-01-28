#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import gym
import random
from collections import defaultdict
import sys
import json
from plot_utils import plot_policy, plot_win_rate


# In[3]:


env = gym.make('Blackjack-v1')
state = env.reset()
action_space = env.action_space.n


# In[4]:


# mapping states to 2 actions (hit or take)
# Building a Q-table

#hyperparameters
num_episodes = 1000000
epsilon = 1
epsilon_min = 0.05 # epsilon greedy strategy we don't always just want exploitaition we also want exploration
alpha = 0.03
gamma = 1


# In[5]:


def monte_carlo(num_episodes, epsilon, epsilon_min, alpha, gamma): 
    
    # defining the Q fucntion 
    q = defaultdict(lambda: np.zeros(2))
    rewards_all_episodes = []
    
    # generating all of the episodes 
    for episode in range(1, num_episodes + 1): 
        epsilon = max(epsilon_min, epsilon * 0.99999)
        
        if episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(episode, num_episodes), end="")
            sys.stdout.flush()
        
        # extracting states, actions, rewards from generated episodes 
        experience = generate_episode(q, epsilon)
        states, actions, rewards = zip(*experience)
        rewards = np.array(rewards)
        rewards_all_episodes.append(sum(rewards))
        
        
        # looping over all of the timesteps 
        for i, state in enumerate(states):
            discounts = np.array([gamma ** i for i in range(len(rewards[i: ]))])
            returns = sum(rewards[i: ] * discounts)
            
            # updating the Q_function using the monte carlo constant alpha update rule 
            q[state][actions[i]] += alpha * (returns - q[state][actions[i]])
            policy = dict((state, np.argmax(q_value)) for state, q_value in q.items())
        
    return q, policy, rewards_all_episodes


# In[6]:


def generate_episode(q,epsilon):
    state = env.reset()
    episode = []
    
    while True :
        action = epsilon_greedy_policy(q,state,epsilon)
        next_state,reward,done,info = env.step(action)
        state = next_state
        episode.append((state,action,reward))
        
        if done == True:
            break
    return episode


# In[7]:


def epsilon_greedy_policy(q,state,epsilon):
    probabilities = np.zeros(2)
    optimal_action = np.argmax(q[state])
    sub_optimal_action = np.abs(optimal_action - 1)
    probabilities[optimal_action] = 1 - epsilon + (epsilon /2)
    probabilities[sub_optimal_action] = epsilon / 2
    
    #choose an acction according to the probabilities
    action = np.random.choice(np.arange(2),p=probabilities)
    return action


# In[ ]:


q, policy, rewards_all_episodes = monte_carlo(num_episodes, epsilon, epsilon_min, alpha, gamma)


# In[ ]:


plot_win_rate(rewards_all_episodes, num_episodes)


# In[ ]:


plot_policy(policy)


# In[ ]:




