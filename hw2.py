import numpy as np

from grid import Grid


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

# Reference: AI TextBook Figure 17.4
def valueIteration(grid, discountFactor, epsilon = 0, maxIter = -1):
    uPrime = grid.deepcopy()
    u = None
    delta, iteration = 0, 0

    while((delta < epsilon * ((1 - discountFactor) / discountFactor)
            or iteration != maxIter)):
            
            u = uPrime.deepcopy()
            delta = 0

            for state in uPrime.getStates():
                if(not (uPrime.isTerminal(state) or uPrime.isBlock(state))):
                    uPrime[state] = u.rewardOf(state) + discountFactor * u.actionWithMaxExpectedValue(state)
                
                    if(abs(uPrime[state] - u[state] > delta)):
                        delta = abs(uPrime[state] - u[state])
            iteration += 1

    return uPrime

# Reference: AI TextBook Figure 17.7
def policyIteration(policyGrid, discountFactor, maxIter=20):
    u = policyGrid.zeroGridUtilities() # Deep copy with u=0
    pi = policyGrid.deepcopy()
    
    count = 0
    while(count != maxIter):
        u = policyEvaluation(pi, u, discountFactor)
     
        for state in pi.getStates():

            if(not (u.isTerminal(state) or u.isBlock(state))):
                givenAction = u.actions(state)[pi[state]]
                
                if(u.actionWithMaxExpectedValue(state) > u.expectedValueForAction(givenAction, state, u)):
                    pi[state] = u.actionWithMaxExpectedValue(state, argmax=True)

        count +=1

    return pi, u

def policyEvaluation(pi, u, discountFactor):
    uPrime = u.deepcopy()

    for state in u.getStates():
        if(not (uPrime.isTerminal(state) or uPrime.isBlock(state))):
            givenAction = u.actions(state)[pi[state]]
            uPrime[state] = u.rewardOf(state) + discountFactor * u.expectedValueForAction(givenAction, state, u)

    return uPrime
        
def findPolicies(grid):
    policyGrid = grid.deepcopy()

    for state in policyGrid.getStates():

        if(not (policyGrid.isTerminal(state) or policyGrid.isBlock(state))):
                # AI TextBook equation 17.4
                action = grid.actionWithMaxExpectedValue(state, argmax=True)
                
                policyGrid[state] = action

    return policyGrid

def setPolicyToAll(grid, policy):
    for state in grid.getStates():
        if(not (grid.isTerminal(state) or grid.isBlock(state))):
            grid[state] = policy

# Q-Learning imp

# Reference: Recitation Slide Figure 11.10
def qLearning(grid, discountFactor, learningRate, epsilon, maxIter=1):
    q = initializeQ(grid, value=0) # Example Item q(s,a) -> q[((sx,sy),(ax,ay))]
    state = grid.getStartingState()

    count = 0 
    while(count != maxIter):
        action = decideToAction(grid, q, state, epsilon)
        statePrime = grid.executeAction(state, action)
  
        q[(state,action)] = (1 - learningRate) * q[(state,action)] + learningRate * (grid.qRewardOf(statePrime) + qMax(q, statePrime))
        state = statePrime
        
        if(grid.isTerminal(state)):
            state = grid.getStartingState()

        count += 1

    return q

def initializeQ(grid, value=0):
    q = {}
    actions = grid.actions((0,0)).keys() # ["N", "S", "W", "E"]

    for state in grid.getStates():
        for action in actions:
            # q[((0,0), "W")]
            q[(state, action)] = 0
    return q
        
def decideToAction(grid, q, state, epsilon):
    r = np.random.rand()

    if(r < epsilon):
        # Decide Exploration Direction
        directions = ["N", "S", "W", "E"]   # [U, D, R, L] -> [N, S, W, E]
        direction = np.random.randint(0,4)
        
        return directions[direction]
    
    return qMax(q, state, argMax=True)

def qMax(q, state, argMax=False):
    directions = ["N", "S", "W", "E"]
    r = np.random.randint(0,4)
    
    # Selecting random max
    argMaxStart = directions[r]
    maxStart = q[(state, argMaxStart)]

    for stateAction, qValue in q.items(): # stateAction = q[s,a] -> q[((sx,sy), "W")]
        if(stateAction[0] == state and qValue > maxStart):
            argMaxStart = stateAction[1]
            maxStart = qValue

    if(argMax == True):
        return argMaxStart
    return maxStart

def qValueToUtilities(grid, q):
    uGrid = grid.deepcopy()

    for state in uGrid.getStates():
        if(not (uGrid.isTerminal(state) or uGrid.isBlock(state))):
            qValue = qMax(q, state, argMax=True)
            s= qMax(q, state, argMax=False)
            uGrid[state] = qValue

    return uGrid

    
def main():


    # grid = createHomeworkGrid(r = -0.04, p = 0.8)


    # valueIterationResult = valueIteration(grid, discountFactor=1, maxIter=1)
    # print(valueIterationResult)

    # grid = sampleGrid0(r=-3)


    # valueIterationResult = valueIteration(grid, discountFactor=1, maxIter=100)
    # print(valueIterationResult)

    # policyGrid = findPolicies(valueIterationResult)
    # print(policyGrid)

    # print(policyGrid.zeroGridUtilities())

    grid = sampleGrid0(r = -3, startState=(3,2))
    # setPolicyToAll(grid, "N")

    # pi = policyIteration(grid, 1, maxIter=100000)
    # print(pi[0])

    # q



    print(grid)
    # print(pi[0])
    # print(pi[1])
    np.random.seed(62)


    # for i in range(50):
    #     print(grid.executeAction((0,0), "W"))

    q = initializeQ(grid)
    a = len(q.keys())
    print(a)
    print(q[((0,0), "S")])
    q[((0,0), "S")] = 10

    q = qLearning(grid, 0.8, 0.1, 0.4, maxIter=100000)
    print(q)
    state = qMax(q,(0,0),argMax=True)
    print(state)
    uGrid = qValueToUtilities(grid, q)
    print(uGrid)
    # armaxq policy
    # print(policy)

main()