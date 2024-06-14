import math

class Node():
    """This class is used for a star, mainly to store heuristic cost etc."""
    g = h = f = 0

    def __init__(self, parent=None, position = None) -> None:
        self.parent = parent #parent node in a star
        self.position = position
        self.g = 0 #distance to start
        self.h = 0 #estimated distance to goal
        self.f = 0 #heuristic cost
    
    def getHeuristicCost(self) -> float:
        """Get heurisitc cost of this node."""
        return self.f
    
    def getParent(self):
        """Returns parent node."""
        return self.parent
    
    def getPosition(self):
        "Returns position of the node."
        return self.position
    
    def __eq__(self, other):
        return self.position == other.position


def aStarPath(width:int , height: int, start: tuple, end:tuple, distanceMetric: bool):
    """Retuns a path generated by A-Star."""
    #print(start, end)
    startNode = Node(None, start)
    endNode = Node(None, end)
    
    #Init open and closed list
    openList = []
    closedList = []

    #Append start node to open list
    openList.append(startNode)

    #Loop until end node is found
    while(len(openList)) > 0:
        #Get current node
        openList.sort(key = Node.getHeuristicCost)
        currentNode: Node = openList.pop(0)

        #Append currently treated node to closed list
        closedList.append(currentNode)

        #This if statement uses the __eq__ function defined in Node class
        #Check if we found goal
        if currentNode == endNode:
            path = []
            current = currentNode
            while current is not None:
                path.append(current.getPosition())
                current = current.getParent()
            path.reverse()
            #print(f"A-Star-Path from {start} to {end}:\n{path}")
            return path
                
        #Generate possible successors
        children:list[Node] = []
        #This is implicitly a 4 neighborhood
        for newPos in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nodePosition = ((currentNode.getPosition()[0] + newPos[0], currentNode.getPosition()[1] + newPos[1]))
            #CAREFUL THIS HAS TO CHECK FOR CORRECT DIMENSIONS!!!!!!!!!!!
            if nodePosition[0] > width or nodePosition[0] < 0 or nodePosition[1] > height or nodePosition[1] < 0:
                continue #Skips illegal children
            
            #Create new child node and append it to the list of children
            newNode = Node(currentNode, nodePosition)
            children.append(newNode)

            #Loop through children
            for child in children:
                if child in closedList:
                    continue
                else:
                    child.g = currentNode.g + 1
                    #This is Manhattan distance
                    if distanceMetric:
                        child.h = abs(child.position[0] - endNode.position[0]) + abs(child.position[1] - endNode.position[1])
                    else:
                        child.h = math.sqrt(math.pow((child.position[0] - endNode.position[0]), 2)+ math.pow((child.position[1] - endNode.position[1]), 2))
                    #TODO: Insert energy here later as weighted sum
                    child.f = (child.g + child.h)

                    #If child is already in open list
                    if child in openList:
                        openNode = openList[openList.index(child)]
                        if child.f < openNode.f:
                            #If openList contains node with worse heuristic cost swap it
                            openList.pop(openList.index(openNode))
                            openList.append(child)
                    else:
                        openList.append(child)