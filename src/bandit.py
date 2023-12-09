import numpy as np

class Bandit:
    def __init__(self, k, distribution):
        self.nr_arms = k
        self.distribution = distribution
        if distribution == "gaussian":
            self.__set_gaussian()
        elif distribution == "bernoulli":
            self.__set_bernoulli()

    def __set_gaussian(self):
        self.means = np.random.normal(size = self.nr_arms) # Standart gaussian distribution with mean = 0, sd = 1 and centered around a random value between -1 and 1

    def __set_bernoulli(self):
        self.probabilities = np.random.uniform(0, 1, self.nr_arms) # Probability from 0 to 1 representing a chance of success

    # Pulls the selected arm in the bandit to return a reward based on arm distribution
    def pull_arm(self, arm_nr):
        if self.distribution == "gaussian":
            return np.random.normal(loc=self.means[arm_nr])
        elif self.distribution == "bernoulli":
            return np.random.binomial(n=1, p=self.probabilities[arm_nr])