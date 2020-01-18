from grid import Grid

# Reference: AI TextBook Figure 17.4
def valueIteration(grid, discountFactor, epsilon = 0, maxIter = 0):
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
                
                    if(abs(uPrime[state] - u[state]) > delta):
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