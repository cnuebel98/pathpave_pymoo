import numpy as np
def findGreedyPath(width, height, start, end):
        """Finds path, that in every step decreases distance to goal"""
        #print(f"Start: {start}, End: {end}")
        path = []
        currentPos = start[0]
        path.append(start)
        while currentPos != end[0]:
            possibleMoves = getPossibleMoves(width, height, currentPos)
            distancesToGoal = [abs(x[0] - end[0][0]) + abs(x[1] - end[0][1]) for x in possibleMoves]
            nextPos = possibleMoves[np.argmin(distancesToGoal)]

            validShiftingPositions = getValidShifting(width, height, currentPos, nextPos)
            path.append((nextPos, validShiftingPositions[np.random.randint(len(validShiftingPositions))]))
            currentPos = nextPos
        # Remove last element since it is the existing end point

        return path 

def getPossibleMoves(width, height, currentPos) -> list:
        """Gives us a list of possible moves in form of coordinates."""
        possibleMoves = [(currentPos[0] + 1, currentPos[1] + 0),
                         (currentPos[0] + 0, currentPos[1] + 1),
                         (currentPos[0] - 1, currentPos[1] + 0),
                         (currentPos[0] + 0, currentPos[1] - 1)]
        possibleMoves = [
            direction for direction in possibleMoves
            if 0 <= direction[0] < height and 0 <= direction[1] < width and direction != currentPos
            ]
        return possibleMoves
    
def getValidShifting(width, height, currentPos, nextPos)->list[tuple]:
    "Returns list of valid shifting positions"
    possibleShifts = [(nextPos[0] + 1, nextPos[1] + 0),
                     (nextPos[0] + 0, nextPos[1] + 1),
                     (nextPos[0] - 1, nextPos[1] + 0),
                     (nextPos[0] + 0, nextPos[1] - 1)]
    possibleShifts = [
        direction for direction in possibleShifts
        if 0 <= direction[0] < height and 0 <= direction[1] < width and direction != currentPos
        ]
    return possibleShifts 