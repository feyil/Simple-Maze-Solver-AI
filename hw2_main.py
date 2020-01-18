from grid import Grid
from mdp_value_policy import *
from q_learning import *

def createHomeworkGrid(r = 0, p = 1, startState=(0,2)):
    grid = Grid(4,4, reward=r, pTransition=p, startingState=startState)

    #Adding Terminals
    grid[(3,0)] = 1
    grid.addTerminal((3,0))

    grid[(1,3)] = 1
    grid.addTerminal((1,3))

    grid[(2,3)] = -10
    grid.addTerminal((2,3))

    grid[(3,3)] = 10
    grid.addTerminal((3,3))

    #Adding Blocks
    grid[(1,1)] = "B"

    return grid

def sampleGrid0(r = -3, p = 0.8, startState = (0,2)):
    grid = Grid(4,3,reward=r,pTransition=p, startingState=startState)

    # Adding Terminals
    grid[(3,0)] = 100
    grid.addTerminal((3,0))

    grid[(3,1)] = -100
    grid.addTerminal((3,1))

    # Adding Blocks
    grid[(1,1)] = "B"
    grid.addBlock((1,1))

    return grid

def main_test():
    # Grid creation
    print("<<Grid Generated>>\n")

    grid = sampleGrid0(r = -3, startState=(0,2))
    print(grid)

    # Value Iteration
    print("<<Value Iteration Calculations Started>>\n")
    print(grid)

    valueIterationResult = valueIteration(grid, discountFactor=1, maxIter=100)
    print(valueIterationResult)

    policyGrid = findPolicies(valueIterationResult)
    print(policyGrid)

    print("<<Value Iteration Calculations Finished>>\n")

    # Policy Iteration
    print("<<Policy Iteration Calculations Started>>\n")
    print(grid)

    iterationGrid = grid.zeroGridUtilities()
    setPolicyToAll(iterationGrid, "N")

    pi = policyIteration(iterationGrid, 1, maxIter=100)
    print(pi[0]) # pi
    print(pi[1]) # utilities

    print("<<Policy Iteration Calculations Finished>>\n")

    # Q-learning
    print("<<Q-learning Calculations Started>>\n")
    print(grid)

    grid.setStartingState((3,2))
    
    q = qLearning(grid, discountFactor=0.8, learningRate=0.1, epsilon=0.4, maxIter=100000)

    puGrid = qValueTo(grid, q)
    print(puGrid[0]) # Policy argmaxQ
    print(puGrid[1]) # Utility maxQ

    print("<<Q-learning Calculations Finished>>\n")

def main_hw(VI, PI, QL, startingState, reward, discountFactor, learningRate, epsilon, probability, N, decay=False):
    
    # Grid Creation
    print("<<Grid Generated>>\n")
    grid = createHomeworkGrid(r=reward, p=probability, startState=startingState)
    print(grid)

    # Value Iteration
    if(VI):
        print("<<Value Iteration Calculations Started>>\n")
        print(grid)

        valueIterationResult = valueIteration(grid, discountFactor=discountFactor, maxIter=N)
        print(valueIterationResult)

        policyGrid = findPolicies(valueIterationResult)
        print(policyGrid)

        print("<<Value Iteration Calculations Finished>>\n")
    
    # Policy Iteration
    if(PI):  
        print("<<Policy Iteration Calculations Started>>\n")
        print(grid)

        iterationGrid = grid.zeroGridUtilities()
        setPolicyToAll(iterationGrid, "N") # N->North->Up

        pi = policyIteration(iterationGrid, discountFactor=discountFactor, maxIter=N)
        print(pi[0]) # pi
        print(pi[1]) # utilities

        print("<<Policy Iteration Calculations Finished>>\n")

    if(QL):
        # Q-learning
        print("<<Q-learning Calculations Started>>\n")
        print(grid)

        grid.setStartingState((0,2))
        
        q = qLearning(grid, discountFactor=discountFactor, learningRate=learningRate, epsilon=epsilon, maxIter=N, decay=decay)

        puGrid = qValueTo(grid, q)
        print(puGrid[0]) # Policy argmaxQ
        print(puGrid[1]) # Utility maxQ

        print("<<Q-learning Calculations Finished>>\n")


np.random.seed(62)

# main_test()

# 1a-Run VI and PI algorithms with ​ r:0​ , ​ d:1​ and ​ p:1​ .
parameters1a = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 0, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": 0,            # r
    "discountFactor": 1,    # d
    "probability": 1,       # p
    "learningRate": 0.1,    # a
    "epsilon": 0.1,         # e
    "N": 1               # N
}

# 1b-With ​ e:0​ , ​ a:0.1​ and ​ N:1000​ , after Q-learning
parameters1b = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": 0,            # r
    "discountFactor": 1,    # d
    "probability": 1,       # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 2a-Now, run the same experiments this time with ​ r:-0.01​ .
parameters2a = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 1,    # d
    "probability": 1,       # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 2b-Update the discount factor as ​ d:0.2​
parameters2b = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 0, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.2,    # d
    "probability": 1,       # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 2c-Change the discount factor back to ​ d:1​ . Update rewards as ​ r:5​ .
parameters2c = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 0, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": 5,        # r
    "discountFactor": 1,    # d
    "probability": 1,       # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 2d-Try VI and PI with ​ d:1​ , ​ r:-0.01​ and ​ p:0.5​ .
parameters2d = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 0, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 1,    # d
    "probability": 0.5,     # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 3e-Run you experiments with ​ d:0.9​ , ​ r:-0.01​ , ​ p:0.8​ for VI and PI.
parameters3e = {
    "VI": 1, # 1->to activate, 0->to deactivate
    "PI": 1, # 1->to activate, 0->to deactivate
    "QL": 0, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,    # d
    "probability": 0.8,     # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 3ei-For Q-learning, in the beginning, continue with ​ e:0​ , ​ a:0.1​ and ​ N:1000​ .
parameters3ei = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 1000               # N
}

# 3eii-What happens if you increase ​ N:10000​ ?
parameters3eii = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 0.1,    # a
    "epsilon": 0,           # e
    "N": 10000              # N
}

# 3eiii-Now let’s introduce some exploration possibility for the agent, ​ e:0.1​ .
parameters3eiii = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 0.1,    # a
    "epsilon": 0.1,         # e
    "N": 10000              # N
}

# 3eiv-Now let’s examine the learning rate. How does the performance change if we update it as ​ a:1​ ?
parameters3eiv = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 1,      # a
    "epsilon": 0.1,         # e
    "N": 10000              # N
}

# 3ev-Implement a counter for each state-action pair and increment it everytime that pair is
# experienced. Set the learning rate to be ​ 1/count​ for each update.
parameters3ev = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 1,      # a
    "epsilon": 0.1,         # e
    "N": 10000,             # N
    "decay": True
}

# 3evi-Finally let’s increase the number of experiments ​ N:100000 ​ (with ​ e:0.1​ and ​ decaying
# learning rate in the previous step).
parameters3evi = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 1,      # a
    "epsilon": 0.1,         # e
    "N": 100000,             # N
    "decay": True
}

# 3evii-What is the best parameter set you can come up with, that reach the optimal policy
# in least number of iterations?
parameters3evii = {
    "VI": 0, # 1->to activate, 0->to deactivate
    "PI": 0, # 1->to activate, 0->to deactivate
    "QL": 1, # 1->to activate, 0->to deactivate
    "startingState": (0,2),
    "reward": -0.01,        # r
    "discountFactor": 0.9,  # d
    "probability": 0.8,     # p
    "learningRate": 1,      # a
    "epsilon": 0.1,         # e
    "N": 100000,             # N
    "decay": True
}

main_hw(**parameters3ev)

