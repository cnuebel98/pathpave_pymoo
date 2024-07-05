from pymoo.core.repair import Repair
from aStar import aStarPath
from greedyPath import findGreedyPath, getValidShifting
import numpy as np
class ErrorRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            for j in range(len(X[i])):
                path = X[i][j]
                for coord in path:
                    if path.count(coord) > 1:
                        raise ValueError(f"The coordinate {coord} appeard more than once in the path, suggesting a circle or ineffective movement\nPath: {path}")
        return X

class PathRepair(Repair):
    """Removes all circles and useless movements from path\nChecks for conncted paths."""

    def _do(self, problem, X, **kwargs):
        #X[0][0] gets real path, so X[0] is path list
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = self.checkConnection(self.eliminateCircles(X[i][j]), problem)
        return(X)

    def eliminateCircles(self, path: list) -> list:
        #We get a path with repeating positions
        for coord in path:
            if path.count(coord) > 1:
                indices = [i for i in range(len(path)) if path[i] == coord]
                newPath = path[:indices[0]] + path[indices[-1]:]
                return self.eliminateCircles(newPath)
        else:
            return path

    def checkConnection(self, path: list, problem) -> list:
        """Checks if path is fully connected, if not we use A-Star to fix."""
        for i in range(len(path)-1):
            currentCord = path[i]
            nextCord = path[i+1]
            possibleNextCoords = [
                (currentCord[0]+1 , currentCord[1]+0),
                (currentCord[0]+0 , currentCord[1]+1),
                (currentCord[0]-1 , currentCord[1]+0),
                (currentCord[0]+0 , currentCord[1]-1),
            ]
            if nextCord not in possibleNextCoords:
                width = problem.width
                height = problem.height
                if nextCord[0] > height or nextCord[0] < 0 or nextCord[1] > width or nextCord[1] < 0:
                    raise ValueError(f"Cord {nextCord} is out of bounds")
                else:
                    path = aStarPath(width, height, currentCord, nextCord, True)
        
        return path
    
class DirectionPathRepair(Repair):
    """Removes all circles and useless movements from path\nChecks for conncted paths."""

    def _do(self, problem, X, **kwargs):
        #X[0][0] gets real path, so X[0] is path list
        for i in range(len(X)):
            for j in range(len(X[i])):
                #X[i][j] = self.checkShiftingDirections(self.checkConnection(self.eliminateCircles(X[i][j]), problem), problem)
                X[i][j] = self.checkShiftingDirections(self.eliminateCircles(X[i][j]), problem)
        return(X)

    def eliminateCircles(self, path: list) -> list:
        #We get a path with repeating positions
        coords = [x[0] for x in path]
        for coord in coords:
            if coords.count(coord) > 1:
                indices = [i for i in range(len(coords)) if coords[i] == coord]
                newPath = path[:indices[0]] + path[indices[-1]:]
                
                return self.eliminateCircles(newPath)
        else:
            return path

    def checkConnection(self, path: list, problem) -> list:
        """Checks if path is fully connected, if not we use greedy path to fix."""
        for i in range(len(path)-1):
            currentCord = path[i][0]
            nextCord = path[i+1][0]
            possibleNextCoords = [
                (currentCord[0]+1 , currentCord[1]+0),
                (currentCord[0]+0 , currentCord[1]+1),
                (currentCord[0]-1 , currentCord[1]+0),
                (currentCord[0]+0 , currentCord[1]-1),
            ]
            if nextCord not in possibleNextCoords:
                raise ValueError(f"Path is not connected\n{[x[0] for x in path]}")
                #width = problem.width
                #height = problem.height
                #if nextCord[0] > height or nextCord[0] < 0 or nextCord[1] > width or nextCord[1] < 0:
                #    raise ValueError(f"Cord {nextCord} is out of bounds")
                #else:
                #    path = findGreedyPath(width, height, currentCord, nextCord) 
        return path
    
    def checkShiftingDirections(self, path:list, problem):
        for i in range(len(path)-1):
            currentCoord = path[i][0]
            nextCoord = path[i+1][0]
            shiftingDirection = path[i+1][1]
            if shiftingDirection == currentCoord:
                validShifts = getValidShifting(problem.width, problem.height, currentCoord, nextCoord)
                path[i+1] = (nextCoord, validShifts[np.random.randint(len(validShifts))])
        return path
            