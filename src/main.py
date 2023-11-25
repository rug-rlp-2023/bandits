import numpy as np
from bandit import Bandit
from agent import Agent

if __name__ == "__main__":
    # Initialize experiment hyperparameters
    n = 100 # Number of scenarios
    k = 10 # Number of different arms (choices)

    # Create bandits
    gaussian_bandits = []
    bernoulli_bandits = []
    
    for i in range(n):
        gaussian_bandit = Bandit(k, "gaussian")
        gaussian_bandits.append(gaussian_bandit)

        bernoulli_bandit = Bandit(k, "bernoulli")
        bernoulli_bandits.append(bernoulli_bandit)

    # Create agents
    strategies = ["epsilon greedy"]#, "greedy", "optimistic greedy", "softmax", "upper-confidence bound", "action preference"]

    # Start experiments
    t = 100 # Number of tries for each bandit

    correct_pulls = []
    epsilon_pulls = []
    epsilons = []
    epsilon = 0.01
    while epsilon < 1:
        for strategy in strategies:
            for bandit in gaussian_bandits:
                #make agent with strategy
                agent = Agent(strategy)
                agent.set_bandit(bandit)
                # For T choose an arm based on agent strategy
                for iteration in range(t):
                    agent.choose_arm(epsilon)
                """
                print("---NEXT AGENT---")
                print("strategy: ", agent.strategy)
                print("total reward: ", agent.cumilative_reward)
                print("correct pulls:", agent.arm_pulls[np.argmax(bandit.means)])
                print("arm pull array: ", agent.arm_pulls)
                print("q_value array: ", agent.q_values)
                print("bandit arm means: ", bandit.means)
                """
                epsilon_pulls.append(agent.arm_pulls[np.argmax(bandit.means)])  
        correct_pulls.append(sum(epsilon_pulls)/len(epsilon_pulls)) 
        epsilons.append(epsilon)  
        epsilon += 0.01
    print(correct_pulls, epsilons)
    print("Best epsilon:", correct_pulls[np.argmax(correct_pulls)], epsilons[np.argmax(correct_pulls)])
            
                
                

    