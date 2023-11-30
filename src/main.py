import numpy as np
from bandit import Bandit
from agent import Agent

if __name__ == "__main__":
    # Initialize experiment hyperparameters
    n = 100 # Number of scenarios
    k = 5 # Number of different arms (choices)

    # Create bandits
    gaussian_bandits = []
    bernoulli_bandits = []
    
    for i in range(n):
        gaussian_bandit = Bandit(k, "gaussian")
        gaussian_bandits.append(gaussian_bandit)

        bernoulli_bandit = Bandit(k, "bernoulli")
        bernoulli_bandits.append(bernoulli_bandit)

    # Create agents
    strategies = ["action preference"]#["greedy", "epsilon greedy", "optimistic greedy", "softmax", "UCB", "action preference"]

    # Start experiments
    t = 500 # Number of tries for each bandit
    
    best_pulls_gaussian = 0
    gaussian_assist = 0
    best_pulls_bernoulli = 0
    bernoulli_assist = 0
    best_param_gaus = -1
    best_param_bern = -1

    temp = 2
    while temp > 0:
        for strategy in strategies:
            average_gaussian_reward = 0
            average_bernoulli_reward = 0
            correct_pulls_gaussian = 0
            correct_pulls_bernoulli = 0

            for bandit in gaussian_bandits:
                #make agent with strategy
                agent = Agent(strategy)
                agent.set_bandit(bandit)
                agent.temperature_action = temp
                # For T choose an arm based on agent strategy
                for iteration in range(t):
                    agent.choose_arm()
                """
                print("---NEXT AGENT GAUSSIAN---")
                print("strategy: ", agent.strategy)
                print("total reward: ", agent.cumilative_reward)
                print("correct pulls:", agent.arm_pulls[np.argmax(bandit.means)])
                print("arm pull array: ", agent.arm_pulls)
                print("q_value array: ", agent.q_values)
                print("action preferences:", agent.action_preferences)
                print("bandit arm means: ", bandit.means)
                """
                average_gaussian_reward += agent.cumilative_reward
                correct_pulls_gaussian += agent.arm_pulls[np.argmax(bandit.means)]
            for bandit in bernoulli_bandits:
                #make agent with strategy
                agent = Agent(strategy)
                agent.set_bandit(bandit)
                # For T choose an arm based on agent strategy
                for iteration in range(t):
                    agent.choose_arm()
                """
                print("---NEXT AGENT BERNOULLI---")
                print("strategy: ", agent.strategy)
                print("total reward: ", agent.cumilative_reward)
                print("correct pulls:", agent.arm_pulls[np.argmax(bandit.probabilities)])
                print("arm pull array: ", agent.arm_pulls)
                print("q_value array: ", agent.q_values)
                print("action preferences:", agent.action_preferences)
                print("bandit arm means: ", bandit.probabilities)
                """
                average_bernoulli_reward += agent.cumilative_reward
                correct_pulls_bernoulli += agent.arm_pulls[np.argmax(bandit.probabilities)]
            """
            print(strategy, "agent: ")
            print("\tAVG reward gaussian =", average_gaussian_reward / n)
            print("\tCorrect pulls gaussian =", correct_pulls_gaussian * 100 / t / n, "%")
            print("\tAVG reward bernoulli =", average_bernoulli_reward / n)
            print("\tCorrect pulls bernoulli =", correct_pulls_bernoulli * 100 / t / n, "%")
            """
            
            if best_pulls_gaussian < correct_pulls_gaussian * 100 / t / n:
                best_pulls_gaussian = correct_pulls_gaussian * 100 / t / n
                bernoulli_assist = correct_pulls_bernoulli * 100 / t / n
                best_param_gaus = temp
            if best_pulls_bernoulli < correct_pulls_bernoulli * 100 / t / n:
                best_pulls_bernoulli = correct_pulls_bernoulli * 100 / t / n
                gaussian_assist = correct_pulls_gaussian * 100 / t / n
                best_param_bern = temp
        
        temp -= 0.1
    print("Gaus:",best_pulls_gaussian, gaussian_assist, "\nBern:", best_pulls_bernoulli, bernoulli_assist, "\nParam val:", best_param_gaus, best_param_bern)


        

                
                

    