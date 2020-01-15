import copy
import math
import numpy as np

class Grid:

    def __init__(self, xGrid, yGrid,reward = 0, uValue = 0, pTransition = 1, startingState = (0,0)):
        self.__grid = self.createGrid(xGrid, yGrid, uValue)
        self.__teriminalStates = []

        self.__xGrid = xGrid
        self.__yGrid = yGrid

        self.__reward = reward

        self.__goTransition = pTransition
        self.__goLeftTransition = (1 - pTransition) / 2
        self.__goRightTransition = (1 - pTransition) / 2

        self.__startingState = startingState
    
    def executeAction(self, state, action):
        # Decide Result of an Action
        r = np.random.rand()

        actions = self.actions(state)

        if(r < 0.8):
            # straight
            return self.actionCheck(actions[action], state)
        elif(r <= 0.9):
            # left
            return self.actionCheck(self.probableState(state, actions[action], "L"), state)
        elif(r <= 1.0):
            # right
            return self.actionCheck(self.probableState(state, actions[action], "R"), state)

    def getStartingState(self):
        return self.__startingState

    def zeroGridUtilities(self):
        zeroGrid = self.deepcopy()

        for state in zeroGrid.getStates():
            if(not (zeroGrid.isTerminal(state) or zeroGrid.isBlock(state))):
                zeroGrid[state] = 0

        return zeroGrid

    def actionWithMaxExpectedValue(self, state, argmax=False):
        actions = self.actions(state)
        exptectedValues = {}

        u = self.__grid

        for compass, action in actions.items():

            expectedValue = self.expectedValueForAction(action, state, u)
            exptectedValues[expectedValue] = compass

        maxExpectedValue = max(exptectedValues)

        if(argmax):
            # Incase of equality, the preference order is up, down, right, left
            preferenceList = ["N", "S", "E", "W"]

            for preference in preferenceList:
                if(exptectedValues[maxExpectedValue] == preference):
                    return preference # Return one of them N,S,E,W
 
        return maxExpectedValue

    def expectedValueForAction(self, givenAction, givenState, u):
        # Transition probabilities obtained
        p = self.__goTransition
        pLeft = self.__goLeftTransition
        pRight = self.__goRightTransition

        # givenAction applied and result states obtained
        goActionState = self.actionCheck(givenAction, givenState)
        goLeftActionState = self.actionCheck(self.probableState(givenState,givenAction, "L"), givenState)
        goRightActionState = self.actionCheck(self.probableState(givenState, givenAction, "R"), givenState)

        expectedValue = p * u[goActionState] + pLeft * u[goLeftActionState] + pRight * u[goRightActionState]

        return expectedValue

    def actionCheck(self, action, state):
        notValid = not self.isValid(action) # stay in same state
        blocked = self.isBlock(action) # stay in same state 

        if(notValid or blocked):
            return state
        return action

    def actions(self, state):
        stateX, stateY = state

        northAction = (stateX, stateY - 1)
        southAction = (stateX, stateY + 1)

        westAction = (stateX - 1, stateY)
        eastAction = (stateX + 1, stateY)

        return {"N":northAction, "S":southAction, "W":westAction, "E":eastAction}

    def probableState(self, state, action, position):
        actionX, actionY = action
        stateX, stateY = state

        stateY = stateY * -1
        degree = 0
        
        if(position == "L"): # LEFT
            degree = -90
        elif(position == "R"): #RIGHT
            degree = 90

        x = actionX - stateX
        y = actionY + stateY

        xRotated = (x * math.cos(math.radians(degree))) - (y * math.sin(math.radians(degree)))
        yRotated = (x * math.sin(math.radians(degree))) + (y * math.cos(math.radians(degree)))

        xRotated += stateX
        yRotated -= stateY

        return (round(xRotated), round(yRotated))

    def getStates(self):
        return self.__grid.keys()

    def rewardOf(self, state):
        return self.__reward

    def qRewardOf(self, state):
        if(self.isTerminal(state)):
            return self[state]
        return self.__reward

    def createGrid(self, xGrid, yGrid, uValue):
        # Top Left (x,y) = (0, 0)
        grid = {}

        for y in range(yGrid):
            for x in range(xGrid):
                grid[(x, y)] = uValue

        return grid

    def addTerminal(self, state):
        if(self.isValid(state)):
            self.__teriminalStates.append(state)

    def addBlock(self, state):
        if(self.isValid(state)):
            self.__grid[state] = "B"        
    
    def isBlock(self, state):
            return (not self.isValid(state)) or self.__grid[state] == "B"
    
    def isTerminal(self, state):
        for terminal in self.__teriminalStates:
            if(state == terminal):
                return True
        return False

    def isValid(self, state):
        x, y = state
        if((x >= 0 and x < self.__xGrid) and (y >= 0  and y < self.__yGrid)):
            return True
        return False
    
    def deepcopy(self):
        return copy.deepcopy(self)

    def __setitem__(self, state, utility):
        if(self.isValid(state)):
            self.__grid[state] = utility
        else:
            message = "Invalid state {}".format(state)
            raise Exception(message)

    def __getitem__(self, state):
        return self.__grid[state]

    def __str__(self):
        output = ""
        for y in range(self.__yGrid):
            output += "y"+ str(y) + " | "
            for x in range(self.__xGrid):
                state = (x,y)
                utility = self.__grid[state]
                if(self.isTerminal(state)):
                    output += "[" + str(utility) + "]"
                else:
                    output +=str(utility)
                output += "  "
            output += "\n"

        return output

# stateX, stateY = 2, 1

# actionX = 1
# actionY = -1

# x = actionX - stateX
# y = actionY + stateY

# degree = 90

# xRotated = (x * math.cos(math.radians(degree))) - (y * math.sin(math.radians(degree)))
# yRotated = (x * math.sin(math.radians(degree))) + (y * math.cos(math.radians(degree)))

# xRotated += stateX
# yRotated -= stateY

# print(round(xRotated), round(yRotated))
    

# grid = Grid(4,4)


# state = grid.probableState((0,0), (1,0), "R")

# print(state)