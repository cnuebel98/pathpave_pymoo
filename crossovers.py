from pymoo.core.crossover import Crossover
import copy
from aStar import aStarPath
import numpy as np
import random

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
            #print("Hi" + str(individual_list))
            for individual in individual_list:
                gene_sequence = individual[0]

                for gene in gene_sequence:
                    
                    #print("Hi" + str(gene))

                    ...

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
    

class OnePointCrossover(Crossover):
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
            newSecondPath = crossoverIndividuals[1][:cutoffPoint]
            path.reverse()
            newSecondPath += path[:-1]
            newSecondPath += crossoverIndividuals[0][cutoffPoint:]
            newFirstPath = newFirstPath
            newSecondPath = newSecondPath



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
            newFirstPath = newFirstPath
            newSecondPath = newSecondPath

            convFirst = np.empty(1, dtype=object)
            convFirst[:] = [newFirstPath]
            X[0] = convFirst
            convSecond = np.empty(1, dtype=object)
            convSecond[:] = [newSecondPath]
            X[1] = convSecond

        return X


class TwoPointCrossover_old(Crossover):
    def __init__(self, prob_crossover, mapSize: tuple):
        super().__init__(2, 2, prob=prob_crossover)
        self.width, self.height = mapSize

    def _do(self, problem, X, **kwargs):
        #X_crossed = copy.deepcopy(X)
        for i in range(0, len(X), 2):
            parent1 = X[i][0][0]
            parent2 = X[i+1][0][0]
            #print("Parent 1:", parent1)
            #print("Parent 2:", parent2)
            size1 = len(parent1)
            size2 = len(parent2)
            
            if size1 < 3 or size2 < 3:
                print("Path is too short for Two Point Crossover")
                continue

            # Select two crossover points for each parent
            cxpoint1_1 = random.randint(1, size1 - 2)
            cxpoint1_2 = random.randint(cxpoint1_1 + 1, size1 - 1)
            cxpoint2_1 = random.randint(1, size2 - 2)
            cxpoint2_2 = random.randint(cxpoint2_1 + 1, size2 - 1)
            #print("points: ", cxpoint1_1, cxpoint1_2, cxpoint2_1, cxpoint2_2)

            # Create offspring by swapping segments
            offspring1 = parent1[:cxpoint1_1] + parent2[cxpoint2_1:cxpoint2_2] + parent1[cxpoint1_2:]
            offspring2 = parent2[:cxpoint2_1] + parent1[cxpoint1_1:cxpoint1_2] + parent2[cxpoint2_2:]
            #print("Offspring 1 before path connection:", offspring1)
            #print("Offspring 2 before path connection:", offspring2)

            # Connect the path segments
            #offspring1 = self.connect_segments(parent1[:cxpoint1_1], parent2[cxpoint2_1:cxpoint2_2], parent1[cxpoint1_2:], problem)
            #offspring2 = self.connect_segments(parent2[:cxpoint2_1], parent1[cxpoint1_1:cxpoint1_2], parent2[cxpoint2_2:], problem)
            #print("Offspring 1 after path connection:", offspring1)
            #print("Offspring 2 after path connection:", offspring2)
                     
            path1_c1 = self.greedy_path_find(parent1[:cxpoint1_1][-1], parent2[cxpoint2_1:cxpoint2_2][0], problem)
            path2_c1 = self.greedy_path_find(parent2[cxpoint2_1:cxpoint2_2][-1], parent1[cxpoint1_2:][0], problem)
            path1_c2 = self.greedy_path_find(parent2[:cxpoint2_1][-1], parent1[cxpoint1_1:cxpoint1_2][0], problem)
            path2_c2 = self.greedy_path_find(parent1[cxpoint1_1:cxpoint1_2][-1], parent2[cxpoint2_2:][0], problem)
            offspring1 = parent1[:cxpoint1_1] + path1_c1 + parent2[cxpoint2_1:cxpoint2_2] + path2_c1 + parent1[cxpoint1_2:]
            offspring2 = parent2[:cxpoint2_1] + path1_c2 + parent1[cxpoint1_1:cxpoint1_2] + path2_c2 + parent2[cxpoint2_2:]

            #print("Offspring 1 after path connection:", offspring1)

            # Check for consecutive duplicates
            offspring1 = self.delete_consecutive_duplicates(offspring1)
            offspring2 = self.delete_consecutive_duplicates(offspring2)

            # Check for duplicates
            self.has_consecutive_duplicates(offspring1)
            self.has_consecutive_duplicates(offspring2)

            # Replace the parents with the offspring
            convFirst = np.empty(1, dtype=object)
            convFirst[:] = [offspring1]
            X[0] = convFirst
            convSecond = np.empty(1, dtype=object)
            convSecond[:] = [offspring2]
            X[1] = convSecond
            
        return X
    
    def has_consecutive_duplicates(self, path):
        for i in range(len(path) - 1):
            if path[i] == path[i + 1]:
                print("Duplicates found")


    def delete_consecutive_duplicates(self, path):
        if not path:
            return path

        cleaned_path = [path[0]]  # Start with the first element
        for i in range(1, len(path)):
            if path[i] != path[i - 1]:
                cleaned_path.append(path[i])
        
        return cleaned_path
   
    def greedy_path_find(self, start, end, problem):
        """Find a path from start to end using greedy best-first search."""
        if start == end:
            return []

        path = [start]
        current_pos = start

        while current_pos != end:
            row, col = current_pos
            
            # Generate all possible moves (up, down, left, right)
            possible_moves = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            
            # Filter out invalid moves
            valid_moves = [
                move for move in possible_moves
                if 0 <= move[0] < self.height and 0 <= move[1] < self.width
            ]
            
            if not valid_moves:
                break  # No valid moves available

            # Choose the move that minimizes the heuristic (Manhattan distance to the end)
            next_pos = min(valid_moves, key=lambda move: self.manhattan_distance(move, end))
            
            # Add the chosen move to the path
            path.append(next_pos)
            current_pos = next_pos

        if path[-1] == end:
            path.pop()

        return path

    def manhattan_distance(self, point1, point2):
        """Calculate the Manhattan distance between two points."""
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    


    def connect_segments(self, start_segment, middle_segment, end_segment, problem):
        """Connect the three segments to form a valid path."""

        path1 = self.greedy_path_find(start_segment[-1], middle_segment[0], problem)
        path2 = self.greedy_path_find(middle_segment[-1], end_segment[0], problem)

        # Remove consecutive duplicates at segment boundaries
        if start_segment and path1 and start_segment[-1] == path1[0]:
            path1 = path1[1:]
        if path1 and middle_segment and path1[-1] == middle_segment[0]:
            middle_segment = middle_segment[1:]
        if middle_segment and path2 and middle_segment[-1] == path2[0]:
            path2 = path2[1:]
        if path2 and end_segment and path2[-1] == end_segment[0]:
            end_segment = end_segment[1:]

        return start_segment + path1 + middle_segment + path2 + end_segment

        #print("start_segment:", start_segment)
        #print("path1:", path1)
        #print("middle_segment:", middle_segment)
        #print("path2:", path2)
        #print("end_segment:", end_segment)
        #start_segment.pop()
        #middle_segment.pop()

class TwoPointCrossover(Crossover):
    def __init__(self, prob_crossover, mapSize: tuple):
        super().__init__(2, 2, prob=prob_crossover)
        self.width, self.height = mapSize

    def _do(self, problem, X, **kwargs):
        for i in range(0, len(X), 2):
            parent1 = X[i][0][0]
            parent2 = X[i + 1][0][0]
            size1 = len(parent1)
            size2 = len(parent2)
            
            if size1 < 3 or size2 < 3:
                print("Path is too short for Two Point Crossover")
                continue

            # Select two crossover points for each parent
            cxpoint1_1 = random.randint(1, size1 - 2)
            cxpoint1_2 = random.randint(cxpoint1_1 + 1, size1 - 1)
            cxpoint2_1 = random.randint(1, size2 - 2)
            cxpoint2_2 = random.randint(cxpoint2_1 + 1, size2 - 1)

            # Create offspring by swapping segments
            offspring1 = parent1[:cxpoint1_1] + parent2[cxpoint2_1:cxpoint2_2] + parent1[cxpoint1_2:]
            offspring2 = parent2[:cxpoint2_1] + parent1[cxpoint1_1:cxpoint1_2] + parent2[cxpoint2_2:]

            # Connect the path segments
            path1_c1 = self.greedy_path_find(parent1[:cxpoint1_1][-1], parent2[cxpoint2_1:cxpoint2_2][0], problem)
            path2_c1 = self.greedy_path_find(parent2[cxpoint2_1:cxpoint2_2][-1], parent1[cxpoint1_2:][0], problem)
            path1_c2 = self.greedy_path_find(parent2[:cxpoint2_1][-1], parent1[cxpoint1_1:cxpoint1_2][0], problem)
            path2_c2 = self.greedy_path_find(parent1[cxpoint1_1:cxpoint1_2][-1], parent2[cxpoint2_2:][0], problem)
            
            offspring1 = parent1[:cxpoint1_1] + path1_c1 + parent2[cxpoint2_1:cxpoint2_2] + path2_c1 + parent1[cxpoint1_2:]
            offspring2 = parent2[:cxpoint2_1] + path1_c2 + parent1[cxpoint1_1:cxpoint1_2] + path2_c2 + parent2[cxpoint2_2:]

            # Check for consecutive duplicates
            offspring1 = self.delete_consecutive_duplicates(offspring1)
            offspring2 = self.delete_consecutive_duplicates(offspring2)

            # Replace the parents with the offspring
            X[i][0][0] = offspring1
            X[i + 1][0][0] = offspring2
            
        return X
    
    def has_consecutive_duplicates(self, path):
        for i in range(len(path) - 1):
            if path[i] == path[i + 1]:
                print("Duplicates found")

    def delete_consecutive_duplicates(self, path):
        if not path:
            return path

        cleaned_path = [path[0]]  # Start with the first element
        for i in range(1, len(path)):
            if path[i] != path[i - 1]:
                cleaned_path.append(path[i])
        
        return cleaned_path
    
    def greedy_path_find(self, start, end, problem):
        """Find a path from start to end using greedy best-first search."""
        if start == end:
            return []

        path = [start]
        current_pos = start

        while current_pos != end:
            row, col = current_pos
            
            # Generate all possible moves (up, down, left, right)
            possible_moves = [
                (row - 1, col), (row + 1, col),
                (row, col - 1), (row, col + 1)
            ]
            
            # Filter out invalid moves
            valid_moves = [
                move for move in possible_moves
                if 0 <= move[0] < self.height and 0 <= move[1] < self.width
            ]
            
            if not valid_moves:
                break  # No valid moves available

            # Choose the move that minimizes the heuristic (Manhattan distance to the end)
            next_pos = min(valid_moves, key=lambda move: self.manhattan_distance(move, end))
            
            # Add the chosen move to the path
            path.append(next_pos)
            current_pos = next_pos

        if path[-1] == end:
            path.pop()

        return path

    def manhattan_distance(self, point1, point2):
        """Calculate the Manhattan distance between two points."""
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])