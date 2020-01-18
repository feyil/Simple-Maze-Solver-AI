import numpy as np
from grid import Grid

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