import numpy as np

class Agent:
    def __init__(self, strategy):
        # default strategy is greedy
        self.strategy = strategy
        self.cumilative_reward= 0
        
    def set_bandit(self, new_bandit):
        self.initial_value = 0
        self.bandit = new_bandit
        self.q_values = np.full(self.bandit.nr_arms, self.initial_value, dtype = float)
        self.arm_pulls = np.zeros(self.bandit.nr_arms)

    def choose_arm(self, epsilon):
        # GREEDY
        if self.strategy == "greedy":
            max_indices = np.where(self.q_values == np.max(self.q_values))[0]   # Choose the arm with the highest estimated value (Q-value)
            chosen_arm = np.random.choice(max_indices)   # Randomly choose one of the indices
        # EPSILON GREEDY
        elif self.strategy == "epsilon greedy":
            if np.random.rand() < epsilon: # epsilon value
                chosen_arm = np.random.choice(self.bandit.nr_arms)
            else:
                max_indices = np.where(self.q_values == np.max(self.q_values))[0]  
                chosen_arm = np.random.choice(max_indices)   
        reward = self.bandit.pull_arm(chosen_arm)
        self.__update_qvalues(chosen_arm, reward)
    
    def __update_qvalues(self, chosen_arm, reward):
        self.cumilative_reward += reward
        self.arm_pulls[chosen_arm] += 1
        #print(self.q_values)
        #print(self.q_values[chosen_arm])
        #temp = self.q_values[chosen_arm]
        self.q_values[chosen_arm] = self.q_values[chosen_arm] + float(((reward - self.q_values[chosen_arm]) / self.arm_pulls[chosen_arm]))
        #print(self.q_values[chosen_arm],"=",temp,"+","((",reward,"-",temp,")/",self.arm_pulls[chosen_arm],")")
        #print(self.q_values)
