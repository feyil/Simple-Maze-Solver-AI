from grid import Grid
from mdp_value_policy import *
from q_learning import *

def createHomeworkGrid(r = 0, p = 1):
    grid = Grid(4,4, reward=r, pTransition=p)

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

def main():
    pass

np.random.seed(62)

main_test()