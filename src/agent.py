import numpy as np
import math

class Agent:
    def __init__(self, strategy):
        # default strategy is greedy
        self.strategy = strategy
        self.cumilative_reward= 0
        # Strategy Variables
        self.temperature = 0.5
        self.temperature_action = 1
        self.epsilon = 0.3
        self.timesteps = 0
        self.c = 1 
        self.a_prefence_rate = 0.5
        
    def set_bandit(self, new_bandit):
        if self.strategy == "optimistic greedy":
            self.initial_value = 1
        else:
            self.initial_value = 0
        self.bandit = new_bandit
        # Action Preferences
        self.action_preferences = np.zeros(self.bandit.nr_arms, dtype = float)
        # Q values
        self.q_values = np.full(self.bandit.nr_arms, self.initial_value, dtype = float)
        self.arm_pulls = np.zeros(self.bandit.nr_arms)

    def choose_arm(self):
        self.timesteps += 1
        # ACTION PREFERENCES
        if self.strategy == "action preference":
            exponentiated_values = np.exp(self.action_preferences / self.temperature_action)
            probabilities = exponentiated_values / np.sum(exponentiated_values)
            chosen_arm = np.random.choice(len(self.action_preferences), p=probabilities)
        # SOFTMAX
        elif self.strategy == "softmax":
            exponentiated_values = np.exp(self.q_values / self.temperature)
            probabilities = exponentiated_values / np.sum(exponentiated_values)
            chosen_arm = np.random.choice(len(self.q_values), p=probabilities)
        # UCB
        elif self.strategy == "UCB":
            ucb = []
            for arm in range(self.bandit.nr_arms):
                if self.arm_pulls[arm] == 0:
                    ucb.append(9999) #very large value for unexplored arms
                else: 
                    ucb.append(self.q_values[arm] + self.c * np.sqrt(np.log(self.timesteps) / self.arm_pulls[arm]))
            chosen_arm = np.argmax(ucb)
        else: 
            # GREEDY
            max_indices = np.where(self.q_values == np.max(self.q_values))[0]   # Choose the arm with the highest estimated value (Q-value)
            chosen_arm = np.random.choice(max_indices)   # Randomly choose one of the indices
            # EPSILON GREEDY
            if self.strategy == "epsilon greedy" and np.random.rand() < self.epsilon: 
                chosen_arm = np.random.choice(self.bandit.nr_arms)
                    
        reward = self.bandit.pull_arm(chosen_arm)
        self.__update_qvalues(chosen_arm, reward)
    
    def __update_qvalues(self, chosen_arm, reward):
        self.cumilative_reward += reward
        self.arm_pulls[chosen_arm] += 1

        # Updates action preferences
        if self.strategy == "action preference":
            # calculates pi_t(a) for all arms
            exp_preferences = np.exp(self.action_preferences)
            pi_t = exp_preferences / np.sum(exp_preferences)

            for preference in range(self.bandit.nr_arms):
                average_reward = self.cumilative_reward / self.timesteps
                if chosen_arm == preference:
                    self.action_preferences[preference] += self.a_prefence_rate * (reward - average_reward) * (1 - pi_t[preference])
                else:
                    self.action_preferences[preference] -= self.a_prefence_rate * (reward - average_reward) * pi_t[preference]
        # Updates q_values
        else:
            self.q_values[chosen_arm] = self.q_values[chosen_arm] + float(((reward - self.q_values[chosen_arm]) / self.arm_pulls[chosen_arm]))
        
        
