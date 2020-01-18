import copy
import matplotlib.pyplot as plt
import numpy as np
from grid import Grid

# Reference: Recitation Slide Figure 11.10
def qLearning(grid, discountFactor, learningRate, epsilon, maxIter=1, decay=False, log=False):
    q = initializeQ(grid, value=0) # Example Item q(s,a) -> q[((sx,sy),(ax,ay))]
    qCount = initializeQ(grid, value=0) # For decaying
    state = grid.getStartingState()
    
    logList = []

    count = 0 
    while(count != maxIter):
        action = decideToAction(grid, q, state, epsilon)
        statePrime = grid.executeAction(state, action)
  
        if(decay):
            # Increment experience counter
            qCount[(state,action)] += 1

            # decaying learning rate
            learningRate = 1 / qCount[(state, action)]

        q[(state,action)] = (1 - learningRate) * q[(state,action)] + learningRate * (grid.qRewardOf(statePrime) + qMax(q, statePrime))
        state = statePrime

        if(log and count % 100 == 0):
            logList.append(copy.deepcopy(q))

        
        if(grid.isTerminal(state)):
            state = grid.getStartingState()

        count += 1

    return q, logList

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

def qValueTo(grid, q):
    uGrid = grid.deepcopy()
    pGrid = grid.deepcopy()

    for state in uGrid.getStates():
        if(not (uGrid.isTerminal(state) or uGrid.isBlock(state))):
            qValue = qMax(q, state, argMax=False)
            uGrid[state] = qValue
            
            qPolicy = qMax(q, state, argMax=True)
            pGrid[state] = qPolicy
            
    return pGrid, uGrid

def plotErrors(grid, logList, pi):
    pPolicy = pi[0]
    pUtility = pi[1]

    errorPolicyList = []
    errorUtilityList = []
    label = []

    count = 100
    for log in logList:
        puGrid = qValueTo(grid, log)

        qPolicy = puGrid[0]
        qUtility = puGrid[1]

        errorPolicy = findNumberOfPolicyError(qPolicy, pPolicy)
        errorPolicyList.append(errorPolicy)

        errorUtility = calculateUtilityErrors(qUtility, pUtility)
        errorUtilityList.append(errorUtility)

        label.append(count)
        count += 100
   
    plotPolicyError(label, errorPolicyList)
    plotUtilityError(label, errorUtilityList)

def findNumberOfPolicyError(qPolicy, pPolicy):
    policyErrorCount = 0
    for state in qPolicy.getStates():
        if(qPolicy[state] != pPolicy[state]):
            policyErrorCount += 1
    return policyErrorCount

def calculateUtilityErrors(qUtility, pUtility):
    errorList = {}
    for state in qUtility.getStates(): 
        if(not (qUtility.isTerminal(state) or qUtility.isBlock(state))):          
            error = pUtility[state] - qUtility[state]
            errorList[state] = error
    return errorList

def plotPolicyError(trialList, errorList):
    # this is for plotting purpose
    index = np.arange(len(errorList))
    plt.bar(index,errorList)
    plt.xlabel('Trials', fontsize=5)
    plt.ylabel('Pollicy Error', fontsize=5)
    plt.xticks(index, trialList, fontsize=5, rotation=30)
    plt.title('Incorrect Policy Count for Experiment')
    plt.show()

def plotUtilityError(label, errorUtilityList):
    stateLabel = {
        (0,0):"s0",
        (1,0):"s1",
        (2,0):"s2",
        (0,1):"s3",
        (2,1):"s4",
        (3,1):"s5",
        (0,2):"s6",
        (1,2):"s7",
        (2,2):"s8",
        (3,2):"s9",
        (0,3):"s10"
    }
    # style
    plt.style.use('seaborn-darkgrid')
    
    # create a color palette
    palette = plt.get_cmap('Set1')
    
    states = errorUtilityList[0].keys()
    plotDic = {}
    for state in states:
        plotDic[state] = []

    for errorUtility in errorUtilityList:
        for state in states:
            plotDic[state].append(errorUtility[state])
    
    num=0
    for state in plotDic.keys():
        num+=1
        plt.plot(label, plotDic[state], marker='', color=palette(num), linewidth=1, alpha=0.9, label=stateLabel[state])
    
    # Add legend
    plt.legend(loc=2, ncol=2)
    
    # Add titles
    plt.title("Utitlity Errors", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.ylabel("Difference policyValue - qValue")
    plt.xlabel("Trial")
    plt.show()
