# Simle Maze Solver AI

### Introuction

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


#### Setup

