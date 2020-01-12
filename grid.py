import copy
import math

class Grid:

    def __init__(self, xGrid, yGrid,reward = 0, uValue = 0, pTransition = 1):
        self.__grid = self.createGrid(xGrid, yGrid, uValue)
        self.__teriminalStates = []

        self.__xGrid = xGrid
        self.__yGrid = yGrid

        self.__reward = reward

        self.__goTransition = pTransition
        self.__goLeftTransition = (1 - pTransition) / 2
        self.__goRightTransition = (1 - pTransition) / 2
    
    def actionWithMaxExpectedValue(self, state):
        actions = self.actions(state)
        exptectedValues = []

        p = self.__goTransition
        pLeft = self.__goLeftTransition
        pRight = self.__goRightTransition

        u = self.__grid

        def actionCheck(action, state):
            notValid = not self.isValid(action) # stay in same state
            blocked = self.isBlock(action) # stay in same state 

            if(notValid or blocked):
                return state
            return action

        for action in actions:

            goAction = actionCheck(action, state)
            goLeftAction = actionCheck(self.probableState(state,action, "L"), state)
            goRightAction = actionCheck(self.probableState(state, action, "R"), state)

            expectedValue = p * u[goAction] + pLeft * u[goLeftAction] + pRight * u[goRightAction]
            exptectedValues.append(expectedValue)

        return max(exptectedValues)

    def actions(self, state):
        stateX, stateY = state

        northAction = (stateX, stateY - 1)
        southAction = (stateX, stateY + 1)

        westAction = (stateX - 1, stateY)
        eastAction = (stateX + 1, stateY)

        return [northAction, southAction, westAction, eastAction]

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