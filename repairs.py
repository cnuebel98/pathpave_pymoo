from pymoo.core.repair import Repair
from aStar import aStarPath

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
                #X[i][j] = self.checkConnection(X[i][j], problem)
                ...
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
        #print(path)
        for i in range(len(path)-1):
            currentCord = path[i]
            #print(currentCord)
            nextCord = path[i+1]
            #print(nextCord)
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
            