# Simple Maze Solver AI

### Introduction

* This repo includes my solution of the given homework(2/2) in the scope of the Artifical Intelligence(CENG461) course which is given as a technical elective in 2019-2020 Fall semester by Computer Engineering Department at Izmir Institute of Technology.
    
* (*)README.md file uses some parts of the official Homework Doc to better express the purpose of the Homework.

### Problem*

* You are expected to implement Value Iteration (VI) and Policy Iteration (PI) algorithms for a Markov Decision Process (MDP) and the Q-learning algorithm for Reinforcement Learning
assuming the same process but without the knowledge of state transition probabilities for available actions.

* The problem is as the following. An agent is going to explore the environment with the transition
properties given below:

![alt text](https://github.com/feyil/Simple-Maze-Solver-AI/blob/master/screenshots/maze-problem-1.png "maze-problem-1")

“s​ i​ ” indicates each state, “r” stands for the state rewards and “p” stands for the probability of the
agent actually going in the chosen direction.

* Other parameters are:
    d: discount factor
    e (epsilon): exploration probability
    a (alpha): learning rate
    N: number of experiments for Q-learning

* The agent starts exploring from the cell painted with radial gradient. It stays in the same cell if it tries to move into a wall (outer edges of the table) or a block (the cell painted in black). The cells painted in gray are terminal states, where the agent has no available actions and the experiment finishes.

* Your implementation should be flexible to run experiments with different parameter sets. You should print utility values in each VI/PI iteration for each state and to display the final values and policies found as a result for all algorithms. Additionally for Q-learning, you should record and plot the utility and policy errors every 100 experiments, assuming that the VI/PI result is optimal.


### Implementation and Result Showcase

* All the implementation made by me and this project started without any base code. I have used different kind of references to complete my solution. I have tried to add comments to code where when I have used a reference to implement it. The main purpose was to understand the core mechanism behind implemented algorithms. I'm always open to feedback. It is plesure for me to learn from you. 

* I also wanted to note that I heavily used "Artificial Intelligence: A Modern Approach 3rd Edition" book to grasp ideas.

#### Setup

* I like to use python virtualenvwrapper you can look my repo to learn how to use it: https://github.com/feyil/Virtualenv-Virtualenvwrapper-Usage

```bash
$ mkvirtualenv ai-maze -p python3
$ workon ai-maze
(ai-maze)$ pip install numpy
(ai-maze)$ pip install matplotlib
(ai-maze)$ python hw2_main.py
```

* You can also use provided requirements.txt file after you set up the environment for the pip installs.

```bash
(ai-maze)$ pip install -r requirements.txt
```

* You can adjust the parameters to look for different behaviour of the algorithms in the hw2_main.py. Also with log=True you can see each step of the algorihtms with some descriptive plots.

```python

parameters3evii = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 0.1,      # a
    "epsilon": 0.4,         # e
    "N": 10000,             # N
    "decay": True,
    "log": False
}

main_hw(**parameters3evii)

```

* N(North)
* S(South)
* E(East)
* W(West)

![alt text](https://github.com/feyil/Simple-Maze-Solver-AI/blob/master/screenshots/maze-problem-2.png "maze-problem-2")

![alt text](https://github.com/feyil/Simple-Maze-Solver-AI/blob/master/screenshots/maze-problem-3.png "maze-problem-3")