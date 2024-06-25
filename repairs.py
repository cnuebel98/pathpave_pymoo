from pymoo.core.repair import Repair

class errorRepair(Repair):
    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            for j in range(len(X[i])):
                path = X[i][j]
                for coord in path:
                    if path.count(coord) > 1:
                        raise ValueError(f"The coordinate {coord} appeard more than once in the path, suggesting a circle or ineffective movement\nPath: {path}")
        return X

class pathRepair(Repair):
    """Removes all circles and useless movements from path."""

    def _do(self, problem, X, **kwargs):
        #X[0][0] gets real path, so X[0] is path list
        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = repairPath(X[i][j])
        return(X)

def repairPath(path: list) -> list:
    #We get a path with repeating positions
    for coord in path:
        if path.count(coord) > 1:
            indices = [i for i in range(len(path)) if path[i] == coord]
            newPath = path[:indices[0]] + path[indices[-1]:]
            return repairPath(newPath)
    else:
        return path