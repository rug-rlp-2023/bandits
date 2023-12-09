import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from bandit import Bandit
from agent import Agent

if __name__ == "__main__":
    # Initialize experiment hyperparameters
    n = 100 # Number of bandits
    k = 5 # Number of different arms for each bandit (choices)

    # Create bandits
    gaussian_bandits = []
    bernoulli_bandits = []
    
    for i in range(n):
        gaussian_bandit = Bandit(k, "gaussian")
        gaussian_bandits.append(gaussian_bandit)

        bernoulli_bandit = Bandit(k, "bernoulli")
        bernoulli_bandits.append(bernoulli_bandit)

    # Start experiments
    t = 8000 # Number of tries for each bandit
    strategies = ["Greedy", "Epsilon Greedy", "Optimistic Greedy", "Softmax", "UCB", "Action Preference"] # Array of all strategies
    agent_data_table = [] # Stores all raw data gotten from simulations
    agent_nr = 0 # track agents
    gaussian_training_reward = [] # store average training reward progression
    gaussian_training_percentages = [] # store training percentage correct pulls
    bernoulli_training_reward = [] # for both cases
    bernoulli_training_percentages = [] # for both cases

    for strategy in strategies:
        # Simulate gaussian bandit scenarios
        i = 0
        gaussian_rewards = np.zeros((n,t)) # track bandit rewards
        gaussian_percentages = np.zeros((n,t)) # track percentage of correct pulls

        for bandit in gaussian_bandits:
            agent = Agent(strategy)
            agent.set_bandit(bandit)

            # Choose an arm based on agent strategy t number of times
            j = 0
            for iteration in range(1, t+1):
                agent.choose_arm()
                gaussian_rewards[i][j] = agent.current_reward
                gaussian_percentages[i][j] = agent.arm_pulls[np.argmax(bandit.means)] * 100 / iteration
                j += 1
           
            # Save data to table
            agent_data = {
                "Bandit type": "gaussian",
                "Agent Nr": agent_nr,
                "Strategy": agent.strategy,
                "Cumilitive reward": agent.cumilative_reward,
                "Nr of correct pulls": agent.arm_pulls[np.argmax(bandit.means)],
                "Nr of timesteps": t,
                "Nr of bandits": n,
                "Nr of arms" : k,
            }
            agent_data_table.append(agent_data)
            i += 1
        
        # average agent strategy reward trend
        gaussian_training_reward.append(np.mean(gaussian_rewards, axis = 0))
        gaussian_training_percentages.append(np.mean(gaussian_percentages, axis = 0))

        # Simulate bernoulli bandit scenarios
        i = 0
        bernoulli_rewards = np.zeros((n,t)) # bandits, timesteps
        bernoulli_percentages = np.zeros((n,t))

        for bandit in bernoulli_bandits:
            agent = Agent(strategy)
            agent.set_bandit(bandit)
            # Choose an arm based on agent strategy t number of times
            j = 0
            for iteration in range(1, t+1):
                agent.choose_arm()
                bernoulli_rewards[i][j] = agent.current_reward
                bernoulli_percentages[i][j] = agent.arm_pulls[np.argmax(bandit.probabilities)] * 100 / iteration
                j += 1
            
            # Save data to table
            agent_data = {
                "Bandit type": "bernoulli",
                "Agent Nr": agent_nr,
                "Strategy": agent.strategy,
                "Cumilitive reward": agent.cumilative_reward,
                "Nr of correct pulls": agent.arm_pulls[np.argmax(bandit.probabilities)],
                "Nr of timesteps": t,
                "Nr of bandits": n,
                "Nr of arms" : k,
            }
            agent_data_table.append(agent_data)
            i += 1

        # Store average bernoulli reward trend  and percentage of correct pulls for agent
        bernoulli_training_reward.append(np.mean(bernoulli_rewards, axis = 0))
        bernoulli_training_percentages.append(np.mean(bernoulli_percentages, axis = 0))

        agent_nr += 1

    # CREATE PLOTS
    # Plot each strategy's average gaussian trend
    plt.figure(figsize=(10, 6))

    plots = []
    for i, strategy in enumerate(strategies):
        plot = plt.scatter(np.arange(1, len(gaussian_training_reward[i]) + 1), gaussian_training_reward[i], label=strategy, s=10)
        plots.append(plot)

    plt.legend(handles = plots)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Rewards During Learning')
    plt.title('Average Reward Trend Across Gaussian Bandits by Time-Step')
    plt.grid(True)
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_GaussianRewards.pdf')
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_GaussianRewards.png')

    # gaussian correct percentage pulls
    plt.figure(figsize=(10,10))

    plots = []
    for i, strategy in enumerate(strategies):
        plot = plt.scatter(np.arange(1, len(gaussian_training_percentages[i]) + 1), gaussian_training_percentages[i], label=strategy, s=10)
        plots.append(plot)

    plt.legend(handles = plots)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Percent of Best Bandit Arm Pulls During Learning' )
    plt.title('Average Percentage of Best Pulled Arms Across Gaussian Bandits by Time-Step')
    plt.grid(True)
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_GaussianPercentages.pdf')
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_GaussianPercentages.png')

    # Plot each strategy's bernoulli percentages
    plt.figure(figsize=(10, 10))
    
    plots = []
    for i, strategy in enumerate(strategies):
        plot = plt.scatter(np.arange(1, len(bernoulli_training_percentages[i]) + 1), bernoulli_training_percentages[i], label=strategy, s=10)
        plots.append(plot)

    plt.legend(handles=plots)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Percent of Best Arm Pulls During Learning')
    plt.title('Average Percent of Best Bandit Arm Pulls Across Bernoulli Bandits by Time-Step')
    plt.grid(True)
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_BernoulliPercentages.pdf')
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_BernoulliPercentages.png')

    # Plot each strategy's average bernoulli trend
    plt.figure(figsize=(10, 6))
    
    plots = []
    for i, strategy in enumerate(strategies):
        plot = plt.scatter(np.arange(1, len(bernoulli_training_reward[i]) + 1), bernoulli_training_reward[i], label=strategy, s=10)
        plots.append(plot)

    plt.legend(handles=plots)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Rewards During Learning')
    plt.title('Average Reward Trend Across Bernoulli Bandits by Time-Step')
    plt.grid(True)
    plt.savefig('C:\\Users\\User\\Desktop\\bandits\\data\\plot_BernoulliRewards.pdf')

    # WRITE DATA
    # Specify the file path relative to the project root
    csv_file_path = "C:\\Users\\User\\Desktop\\bandits\\data\\data.csv"

    # Ensure that the target directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    # write
    with open(csv_file_path, mode = 'a+', newline = '') as file:
        fieldnames = ["Bandit type", "Agent Nr", "Strategy", "Cumilitive reward", "Nr of correct pulls", "Nr of timesteps", "Nr of bandits", "Nr of arms"]
        writer = csv.DictWriter(file, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(agent_data_table)
    

        

                
                

    