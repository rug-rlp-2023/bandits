# Assignment 1: Multi-Armed Bandits

**Deadline: 13 December 2023 23:59**

_This assignment serves as a preparation for the final, open project. It is highly advised you adhere to best software engineering practices since the final project may be significantly bigger than this one._

## Instructions

In this first assignment, you will implement two versions of the multi-armed bandit problem in Python.
At the end of the assignment, you will have to submit a report explaining your results (on Brightspace)
as well as the code used for your experiments (on GitHub). The multi-armed bandit problems that you
have to implement are based on the second theoretical lecture and are the following:

- **The Gaussian bandit**: a multi-armed bandit in which the reward obtained from each action is
  sampled from a normal distribution
- **The Bernoulli bandit**: a multi-armed bandit in which the reward obtained from each action is
  sampled from a Bernoulli distribution (each arm has probability $p$ to return 1 and $1−p$ probability
  to return 0)

As seen throughout the second theoretical lecture, the goal of the agent for each bandit problem is
to learn an optimal policy $\pi^*$, i.e. the action that brings the maximum reward. This goal is reached
through learning. You have to create a set of $N$ randomly generated $k$-armed bandit problems for both
bandit scenarios, where both $k$ and $N$ are parameters of your choice. For each of those problems, you
will then train an agent with the different exploration methods that we have seen in lecture 2. These
methods are the following:

- Greedy and $\epsilon$-greedy
- Optimistic initial values
- Softmax policy
- Upper-Confidence Bound
- Action Preferences

You will perform one experiment for each exploration method. Each experiment will consist of a
number of training steps $T$. At the end of each training run, we expect your agent to have learned to
recognize the action (or actions) that allow it to obtain the maximum possible reward. Furthermore,
each experiment will be repeated for a certain number of times (for example $N = 1000$). You will
measure the learning performance of the agent by monitoring the average reward it obtains, as well as
the percentage of times the agent chooses the best action. Note that you might have to fine-tune the
hyperparameters that govern the learning process for every experiment, which could change across the
different exploration methods.

## Report

You should describe and present your experiments in a written report, which should not be longer than 4
pages, and follow the LATEX template you can find [here](https://resource-cms.springernature.com/springer-cms/rest/v1/content/19238648/data/v7). Your report has to include:

- A brief description of each algorithm you have used, with their respective equations
- A description of the experimental setup used, the definition of the reward functions as well as which
  hyperparameters you chose for each algorithm
- For both problems, you will plot your results in two clear figures: one containing the trend of the
  rewards during learning and one containing the percentage of times the best action is selected
- A section where you compare and discuss the performance of the different exploration algorithms.
  If there are one or more algorithms with significantly better performance than all the others, explain
  why

## Code

Your code should be clean, well-written and documented. You should use Python 3.

### Structure

The code repository comes with a pre-defined structure to ease your efforts. For now, it is highly advised you stick to it. However, if you think your idea for code structure is better, go for it. However, all the changes made to the repository structure need to be comprehensively argued for in the Pull Request description (more on that later). And no worries, you will have a lot of freedom during the final project :) The defined structure works best if you make use of OOP in Python. This is however not required.

- 📁 **`src/`** : Your code for experiments should be here
  - 🐍 `__init.py__` : Marks directory as Python package. DO NOT TOUCH!
  - 🐍 `agent.py` : Your implementation of agents.
  - 🐍 `bandit.py` : Your implementation of bandits.
  - 🐍 `main.py` : The starting point of your program.
- 📁 **`data/`** : All of your results should be here: `.csv` files, figures, etc.
- 📁 **`analysis/`** : _(optional)_ Scripts and notebooks you use to analyze the results.
- 🐍 `setup.py` : Sets up the project. DO NOT TOUCH!
- 📄 `requirements.txt` : List of all the required libraries used in your project.
- 📄 `.gitignore` : Files to be ignored by `git`. DO NOT TOUCH!

Let's dive into a bit more detail.

#### `src/`

Your implementation of the bandits as well as your RL agents should be here. Since you should implement multiple types of bandits and agents, feel free to create more files in this directory. The `main.py` is a starting point of your program and it should generate all the (raw) results of your experiments. This means that it does not necessarily need to generate all the figures you use in the report. However, it should generate all the data you will use to generate these figures, for example `.csv` files containing the training results of different agent/bandit combinations. Remember, that these results should be stored in the **`data/`** directory.

#### `data/`

All your results should be here. This includes all the figures you use in your report as well as the files containing the raw data, for example `.csv`. Note that this directory is included in the `.gitignore`. This means that its contents should not be pushed to GitHub.

#### `analysis/`

If you decide to perform some additional analysis of your results or generate figures outside of your `main.py` script, put the code you used here. This directory should preferebly contain Jupyter Notebooks (`.ipynb` files) that load raw results from `data/` directory, analyze them and generate appropriate figures.

### Replication

You need to make sure the results are replicable. Achieving this is a bit different depending on the tools you use. Make sure early on that this step works well on your machine. Otherwise, we might not be able to verify that your code runs correctly.

#### `pip`

Follow these steps if you are using the `pip` package manager.

1. Download the [pipreqs](https://pypi.org/project/pipreqs/) package, using pip: `pip3 install pipreqs`
2. Generate the list of the libraries you use in your implementation: `pipreqs --force --savepath requirements.txt src/`
3. Verify that your project works:
   1. Run the setup script: `pip3 install .`
   2. Run your implementation: `python3 src/main.py`
   3. Your program should run without any problem and put the generated (raw) results in the `data/` directory.

#### `conda`

If you are using the `conda` package manager, you should first install `pip` inside of your `conda` environment: `conda install pip`. Follow the same steps as in the `pip` case. If you run into problems, consult this wonderful [StackOverflow question](https://stackoverflow.com/questions/41060382/using-pip-to-install-packages-to-anaconda-environment). If the issues, persist, contact the TA.

#### Sample

After following the above steps, your `requirements.txt` file could look somewhat like this:

```
matplotlib==3.6.2
numpy==1.23.5
pandas==1.5.1
```

## Submission

You need to submit both the code and the report. The report should be submitted through Brightspace in a standard manner. The code should be submitted on GitHub by opening a Pull Request from the branch you were working on to the `submission` branch. Before you do so, make sure that:

1. Your experiments are replicable, as described above.
2. Your `main.py` runs **all** experiments.
3. The `data/` directory stores all the figures and results from your experiments.

There are automated checks that verify that your submission is correct:

1. Deadline - checks that the last commit in a PR was made before the deadline
2. Reproducibility - downloads libraries included in `requirements.txt` and runs `python3 src/main.py`. If your code does not throw any errors, it will be marked as reproducible.

For more information regarding the submission system, consult the slides of _Coding Assignments Instructions_ available on Brightpsace (Content->Assignments->Coding Assignments Instructions). If your questions are still unanswered, do not hesitate to contact the TA.

## Grading Criteria

Each assignment will be graded based on the following evaluation criteria for which you will receive a grade between $1$ and $10$:

| Criterion    | Description                                                                                                                                                                                                                                                                                                                                                                                                           | Weight |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Presentation | This criterion will consider how well the overall report looks like: "does it contain spelling mistakes?"; "is the writing redundant or not-concise?"; "do you respect the given LaTeX template and the maximum page limit?", "is the report well structured?"; etc ..                                                                                                                                                | 10%    |
| Clarity      | This criterion considers how understandable and precise your final report is: "are the different exploration strategies explained correctly?"; "are the different bandit problems described in detail?"; "is every variable used in an equation well defined?"; "is the overall experimental setup well described, and does it contain enough details to allow reproducibility?"                                      | 40%    |
| Results      | It is crucial that the report shows proof that you successfully trained an agent on both bandit problems mentioned beforehand. This means that you will have to report learning curves, possibly averaged over different training runs, that allow you to statistically assess the agents' performance. These results have to be reported in clear plots with understandable and readable captions, axes, and titles. | 25%    |
| Code Quality | You should follow the guidelines described in the Code section. Your code should be well-formatted and adhere to the best software engineering practices, such as proper abstraction and re-usability.                                                                                                                                                                                                                | 25%    |

Your code will be reviewed on GitHub. You will receive comments on your code inside of a Pull Request. Some of these comments may include questions regarding your design. Feel free to argue your choices there! Note that this can only increase your grade as a part of the _Awesome Code_ criterion.

## Plagiarism

We know that there are a lot of tutorials on Reinforcement Learning online. If you find them useful, please learn from them, but keep in mind that some of the most popular tutorials have mistakes in them 😲. Further, if we suspect that a group plagiarised and copied someone else's approach without proper understanding, we will schedule a meeting with them to verify their integrity or immediately fail them if the plagiarism is obvious.

## Tips and Resources

Here are a couple of hints and resources that might help you with in this assignment:

1. To help you out with technical writing, check out these papers for inspiration. Reading real
   scientific papers can help you out with using correct nomenclature and ensuring a clear structure.
   In particular, you can draw inspiration as to how complex concepts and formulas are introduced
   and explained.

   a. Technical Report on implementing RL algorithms in CartPole environment - https://arxiv.org/pdf/2006.04938.pdf

   b. Paper summarising usage of RL in Chess - https://page.mi.fu-berlin.de/block/concibe2008.pdf

2. If you have duplicate code in multiple places, it’s probably a bad sign. Maybe you should try it to
   group that functionality in a seperate function?
3. The agent should be able to learn using different types of algorithms. Maybe there is a way to
   make these algorithms easily swappable?
4. The agent’s implementation is agnostic with regards to the type of bandit it’s interacting with.
   Maybe that can help with reducing duplicate code.
5. Type hinting is not required, but it can help your partner understand your code - https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
6. Git workshop by Cover - https://studysupport.svcover.nl/?a=1
7. YouTube Git tutorial - https://www.youtube.com/watch?v=RGOj5yH7evk
8. OOP in Python - https://www.youtube.com/watch?v=JeznW_7DlB0
9. How to document Python? - https://www.datacamp.com/tutorial/docstrings-python4

## Questions and help

If you are struggling with one part of the assignment, you're probably not alone. That's why we want to create a small FAQ throughout the next couple of weeks. In case of a question, raise an issue in the original, template repository: https://github.com/rug-rlp-2023/bandits. We will answer your questions there, so that there are no duplicate questions.
