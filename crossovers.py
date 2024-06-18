from pymoo.core.crossover import Crossover
import copy
from aStar import aStarPath
import numpy as np

class CopyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)
    
    def _do(self, problem, X, **kwargs):
        #print("CopyCrossover")
        return copy.deepcopy(X)  # Return deep copies of the parent


class CrossingCrossover(Crossover):
    def __init__(self, prob_crossover):
        super().__init__(2, 2, prob=prob_crossover)
    
    def _do(self, problem, X, **kwargs):
        X_crossed = copy.deepcopy(X)
        #print(X_crossed)
        #print("len von pop: " + str(len(X_crossed)))
        #print(type(X_crossed))

        for individual_list in X_crossed:
            for individual in individual_list:
                gene_sequence = individual[0]
                print("Hi" + str(gene_sequence))



                #for gene in gene_sequence:
                    #print("hi " + str(gene))
                 #   ...
            # list of list with first Individal
            #print("hi1" + str(individual[0]))
            # list with first Individal
            #print("hi2" + str(individual[0][0]))
            # first coordinate of Individual
            #print("hi3" + str(individual[0][0][0]))
            # X coordinate of first coordinate of Individual
            #print("hi4: " + str(individual[0][0][0][0]))
            
        return X_crossed
    

class onePointCrossover(Crossover):
    def __init__(self, prob_crossover, mapSize: tuple):
        super().__init__(2, 2, prob=prob_crossover)
        self.width, self.height = mapSize

    def _do(self, problem, X, **kwargs):
        #print(f"X: {X}")
        crossoverIndividuals = []
        for individual_list in X:
            for individual in individual_list:
                crossoverIndividuals.append(individual[0])
        
        #You have to check length of individuals to set maximum cutoff point
        if len(crossoverIndividuals[0])<=len(crossoverIndividuals[1]):
            maxLength = len(crossoverIndividuals[0])
        else:
            maxLength = len(crossoverIndividuals[1])

        #Randomly get cutoff point
        cutoffPoint = np.random.randint(1, maxLength-1)

        #Get points in each individual
        firstPoint = crossoverIndividuals[0][cutoffPoint]
        secondPoint = crossoverIndividuals[1][cutoffPoint]

        if firstPoint != secondPoint:
            #TODO: Get map parameters here
            path = aStarPath(self.width, self.height, firstPoint, secondPoint, True)
            newFirstPath = crossoverIndividuals[0][:cutoffPoint]
            newFirstPath += path[:-1]
            newFirstPath += crossoverIndividuals[1][cutoffPoint:]
            newFirstPath = self.checkNewPath(newFirstPath)
            newSecondPath = crossoverIndividuals[1][:cutoffPoint]
            path.reverse()
            newSecondPath += path[:-1]
            newSecondPath += crossoverIndividuals[0][cutoffPoint:]
            newSecondPath = self.checkNewPath(newSecondPath)

            #print(f"New first path: {newFirstPath}")
            #print(f"New second path: {newSecondPath}")


            convFirst = np.empty(1, dtype=object)
            convFirst[:] = [newFirstPath]
            X[0] = convFirst
            convSecond = np.empty(1, dtype=object)
            convSecond[:] = [newSecondPath]
            X[1] = convSecond
            #print(X)
        else:
            newFirstPath = crossoverIndividuals[0][:cutoffPoint]
            newFirstPath += crossoverIndividuals[1][cutoffPoint:]
            newSecondPath = crossoverIndividuals[1][:cutoffPoint]
            newSecondPath += crossoverIndividuals[0][cutoffPoint:]
            newFirstPath = self.checkNewPath(newFirstPath)
            newSecondPath = self.checkNewPath(newSecondPath)

            convFirst = np.empty(1, dtype=object)
            convFirst[:] = [newFirstPath]
            X[0] = convFirst
            convSecond = np.empty(1, dtype=object)
            convSecond[:] = [newSecondPath]
            X[1] = convSecond

        return X
    
    def checkNewPath(self, path:list):
        i = 0
        while i+2 < len(path):
            first = path[i]
            second = path[i+2]
            if first == second:
                #print(f"Step 1:{path[i]}, Step 2:{path[i+1]}, Step 3: {path[i+2]} ")
                #print("REPAIRING PATH")
                #This has to be i+1 both times since the popped index is deleted and everything afterwards moved 1 index back
                path.pop(i+1)
                path.pop(i+1)
                #print(f"REPAIRED PATH Step 1:{path[i]}, Step 2:{path[i+1]}, Step 3: {path[i+2]} ")
            i+=1
        return path
