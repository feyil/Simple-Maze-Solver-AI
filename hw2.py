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

def sampleGrid0(r = -3, p = 0.8):
    grid = Grid(4,3,reward=r,pTransition=p)

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

# Incase of equality !!!!!! up, down, right, left

# grid = createHomeworkGrid(r = -0.04, p = 0.8)


# valueIterationResult = valueIteration(grid, discountFactor=1, maxIter=1)
# print(valueIterationResult)

# grid = sampleGrid0(r=-3)


# valueIterationResult = valueIteration(grid, discountFactor=1, maxIter=100)
# print(valueIterationResult)

# policyGrid = findPolicies(valueIterationResult)
# print(policyGrid)

# print(policyGrid.zeroGridUtilities())

grid = sampleGrid0(r = -3)
setPolicyToAll(grid, "N")

pi = policyIteration(grid, 1, maxIter=100)

print(grid)
print(pi[0])
print(pi[1])