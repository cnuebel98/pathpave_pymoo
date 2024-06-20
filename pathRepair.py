def repairPath(path: list) -> list:
    #We get a path with repeating positions
    for coord in path:
        if path.count(coord) > 1:
            indices = [i for i in range(len(path)) if path[i] == coord]
            newPath = path[:indices[0]] + path[indices[-1]:]
            return repairPath(newPath)
    else:
        return path
                
