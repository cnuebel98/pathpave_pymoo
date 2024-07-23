from pymoo.core.mutation import Mutation
import copy
import random
import numpy as np
from greedyPath import getValidShifting

class CopyMutation(Mutation):
    def __init__(self):
        super().__init__(1)  # It takes one individual and returns one mutant
    
    def _do(self, problem, X, **kwargs):
        return copy.deepcopy(X)  # Return a deep copy of the individual

class ChangePartsMutation(Mutation):
    '''This cuts the individual at two parts, and finds a new random way between the two cuts'''
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate
        super().__init__(1) # It takes one individual and returns one mutant

    def _do(self, problem, X, **kwargs):
        # Input: The whole Population
        # Output: The whole Mutated Population
        
        # For each individual in the Population we decide, 
        # if it gets Mutated or not according to the Mutationrate
        X_mut = copy.deepcopy(X)
        for i in range(len(X_mut)):
            if random.random() < self.mutation_rate:

                # Find a random point in the individual to cut it into two
                individual_length = len(X_mut[i][0])
                
                start_point_for_swap = random.randint(1, individual_length-1)
                
                part_one = X_mut[i][0][:start_point_for_swap]
                part_two = X_mut[i][0][start_point_for_swap:]
                
                # To not change the Individual too much, the longer 
                # sequence of the Gene stays and the shorter one is 
                # getting replaced so we either need to find a new
                # way to the start cell or the end cell, depending on the cut

                if len(part_one) >= len(part_two):
                    # keep part one and find new random way to end cell
                    new_part = self._generate_random_path(problem, X_mut[i][0][start_point_for_swap], problem.end)
                    # use AStar to generate the new path
                    #new_part = aStarPath(width=problem.width, height=problem.height, start=X_mut[i][0][start_point_for_swap], end=problem.end, distanceMetric=False)

                    if new_part and part_one[-1] == new_part[0]:
                        new_part = new_part[1:]
                    new_genes = part_one + new_part
                elif len(part_two) > len(part_one):
                    # keep part two and find new random way to start cell
                    new_part = self._generate_random_path(problem, X_mut[i][0][start_point_for_swap], problem.start)

                    # use AStar to generate the new path
                    #new_part = aStarPath(width=problem.width, height=problem.height, start=X_mut[i][0][start_point_for_swap], end=problem.start, distanceMetric=False)

                    # here we have to reverse the newly generated list before appending it to the part_two because we are finding a new way to the start
                    new_part_mirrored = new_part[::-1]
                    
                    # Then we can add this to the new gene set
                    if new_part_mirrored and new_part_mirrored[-1] == part_two[0]:
                        new_part_mirrored = new_part_mirrored[:-1]
                    new_genes = new_part_mirrored + part_two
                    
                X_mut[i][0] = new_genes
        return np.array(X_mut)
    
    def _generate_random_path(self, problem, start, end):
        
        path = [start]
        current_pos = start

        while current_pos != end:
            row, col = current_pos
            # Generate all possible moves (up, down, left, right)
            possible_moves = [
                (row-1, col), (row+1, col), 
                (row, col-1), (row, col+1)
            ]
            
            # Filter out invalid moves
            valid_moves = [
                move for move in possible_moves 
                if 0 <= move[0] < problem.height and 0 <= move[1] < problem.width and move != current_pos
            ]

            # Ensure no consecutive duplicates
            if len(path) > 1:
                last_pos = path[-1]
                valid_moves = [move for move in valid_moves if move != last_pos]
                
            # Ensure no immediate backtracking to the previous position
            if len(path) > 2:
                previous_pos = path[-2]
                valid_moves = [move for move in valid_moves if move != previous_pos]

            # Choose a valid move randomly
            if valid_moves:
                next_pos = random.choice(valid_moves)
                path.append(next_pos)
                current_pos = next_pos
            else:
                break
        
        return path
    
class RectangleMutation(Mutation):
    '''This cuts the individual at two parts, and finds a new random way between the two 
    cuts, within the bounds of the min and max x and y coordinates'''
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate
        super().__init__(1) 
   
    def _do(self, problem, X, **kwargs):
        # Input: The whole Population
        # Output: The whole Mutated Population
        
        # For each individual in the Population we decide, 
        # if it gets Mutated or not according to the Mutationrate
        X_mut = copy.deepcopy(X)
        for i in range(len(X_mut)):
            if random.random() < self.mutation_rate:

                # Find a random point in the individual to cut it into two
                individual_length = len(X_mut[i][0])
                
                # make the cut to the length on 20% of the individual
                # how many genes are 20% of the individual?
                len_of_cut_part = int(0.2*len(X_mut[i][0]))

                x = random.randint(1, individual_length-1-len_of_cut_part)
                y = x + len_of_cut_part
                
                if x < y:
                    start_point_for_swap = x
                    end_point_for_swap = y
                else: 
                    start_point_for_swap = y
                    end_point_for_swap = x

                part_one = X_mut[i][0][:start_point_for_swap]
                part_two = X_mut[i][0][end_point_for_swap:]
                
                y1, x1 = part_one[-1]
                y2, x2 = part_two[0]

                # Calculate width bounds (min and max x coordinates)
                min_x = min(x1, x2)
                max_x = max(x1, x2)
                width_bounds = (min_x, max_x)

                # Calculate height bounds (min and max y coordinates)
                min_y = min(y1, y2)
                max_y = max(y1, y2)
                height_bounds = (max_y, min_y)

                new_part = self.generate_semi_random_path(width_bounds=width_bounds, height_bounds=height_bounds, start=part_one[-1], end=part_two[0])
                
                new_genes = part_one + new_part + part_two

                X_mut[i][0] = new_genes
        return np.array(X_mut)
    
    def generate_semi_random_path(self, width_bounds, height_bounds, start, end):
        
        path = []
        current_pos = start
        #print("curr pos: " + str(current_pos))

        while current_pos != end:
            row, col = current_pos
            # Generate all possible moves (up, down, left, right)
            possible_moves = [
                (row-1, col), (row+1, col), 
                (row, col-1), (row, col+1)
            ]
            
            # Filter out invalid moves
            valid_moves = []
    
            valid_moves = [
                move for move in possible_moves 
                if height_bounds[1] <= move[0] <= height_bounds[0]  and width_bounds[0] <= move[1] <= width_bounds[1] and move != current_pos
            ]
            
            # Choose a valid move randomly
            if valid_moves:
                next_pos = random.choice(valid_moves)
                path.append(next_pos)
                current_pos = next_pos
            else:
                break
        path.pop()
        return path
    
class RadiusSamplingMutation(Mutation):
    '''cuts the individual at 2 points. Find the midpoint between the two cuts. 
    Then samples a random point within a specific radius around the midpoint. 
    Then it finds a greedy new path connectiong the two cutoff points with the sampled point'''
    def __init__(self, mutation_rate, radius, problem):
        self.mutation_rate = mutation_rate
        self.radius = radius
        self.problem = problem
        super().__init__(1)
   
    def _do(self, problem, X, **kwargs):
        X_mut = copy.deepcopy(X)
        
        for i in range(len(X_mut)):
            if random.random() < self.mutation_rate:
                individual_length = len(X_mut[i][0])
                
                len_of_cut_part = int(0.2 * len(X_mut[i][0]))
                x = random.randint(1, individual_length - 1 - len_of_cut_part)
                y = x + len_of_cut_part
                
                if x < y:
                    start_point_for_swap = x
                    end_point_for_swap = y
                else: 
                    start_point_for_swap = y
                    end_point_for_swap = x

                part_one = X_mut[i][0][:start_point_for_swap]
                part_two = X_mut[i][0][end_point_for_swap:]
                
                y1, x1 = part_one[-1]
                y2, x2 = part_two[0]
                
                midpoint = self.find_midpoint(y1, x1, y2, x2)
                sampled_point = self.sample_point_within_radius(midpoint[0], midpoint[1], self.radius, self.problem, X_mut[i][0])
                
                new_part_one = self.greedy_path_find(part_one[-1], sampled_point, self.problem)
                new_part_two = self.greedy_path_find(sampled_point, part_two[0], self.problem)
                
                new_genes = part_one + new_part_one + [sampled_point] + new_part_two + part_two
                X_mut[i][0] = new_genes
                
        return np.array(X_mut)
    
    def find_midpoint(self, y1, x1, y2, x2):
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        return (int(mid_y), int(mid_x))
    
    def sample_point_within_radius(self, mid_y, mid_x, radius, problem, individual):
        while True:
            rand_y = mid_y + random.randint(-radius, radius)
            rand_x = mid_x + random.randint(-radius, radius)
            
            if 0 <= rand_y < problem.height and 0 <= rand_x < problem.width:
                if (rand_y, rand_x) not in individual:
                    return (rand_y, rand_x)
    

    def manhattan_distance(self, point1, point2):
        """Calculate the Manhattan distance between two points."""
        #print("Manhatten")
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


    def greedy_path_find(self, start, end, problem):
        """Find a path from start to end using greedy best-first search."""
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
                if 0 <= move[0] < problem.height and 0 <= move[1] < problem.width
            ]
            
            if not valid_moves:
                break  # No valid moves available

            # Choose the move that minimizes the heuristic (Manhattan distance to the end)
            next_pos = min(valid_moves, key=lambda move: self.manhattan_distance(move, end))
            
            # Add the chosen move to the path
            path.append(next_pos)
            current_pos = next_pos

        # Remove the last point from the path if it's the same as 'end'
        if path[-1] == end:
            path.pop()

        return path

class RadiusDirectionSamplingMutation(Mutation):
    '''cuts the individual at 2 points. Find the midpoint between the two cuts. 
    Then samples a random point within a specific radius around the midpoint. 
    Then it finds a greedy new path connectiong the two cutoff points with the sampled point.
    Afterwards, we add valid shifting directions to fit it to the dicretional shifting approach.'''
    def __init__(self, mutation_rate, radius, problem):
        self.mutation_rate = mutation_rate
        self.radius = radius
        self.problem = problem
        super().__init__(1)
   
    def _do(self, problem, X, **kwargs):
        X_mut = copy.deepcopy(X)
        #X_mut[1][0][0] gives us the tuples of one gene
        
        for i in range(len(X_mut)):
            if random.random() < self.mutation_rate:
                individual_length = len(X_mut[i][0])
                
                len_of_cut_part = int(0.2 * len(X_mut[i][0]))
                x = random.randint(1, individual_length - 1 - len_of_cut_part)
                y = x + len_of_cut_part
                
                if x < y:
                    start_point_for_swap = x
                    end_point_for_swap = y
                else: 
                    start_point_for_swap = y
                    end_point_for_swap = x

                part_one = X_mut[i][0][:start_point_for_swap]
                part_two = X_mut[i][0][end_point_for_swap:]
                
                y1, x1 = part_one[-1][0]
                y2, x2 = part_two[0][0]

                midpoint = self.find_midpoint(y1, x1, y2, x2)
                sampled_point = self.sample_point_within_radius(midpoint[0], midpoint[1], self.radius, self.problem, X_mut[i][0])
                
                new_part_one = self.greedy_path_find(part_one[-1][0], sampled_point, self.problem)
                new_part_two = self.greedy_path_find(sampled_point, part_two[0][0], self.problem)
                
                #Idea is to sample path without directions and then add valid shifting direction afterwards
                newCombinedParts = new_part_one + [sampled_point] + new_part_two
                tmp = []

                #Get shifting for first coord in newCombinedParts
                validFristCoords = getValidShifting(problem.width, problem.height, part_one[-1], newCombinedParts[0])
                tmp.append(((newCombinedParts[0]), validFristCoords[random.randint(0, len(validFristCoords)-1)]))

                #Get shifting for rest of the parts besides last new coord
                for j in range(len(newCombinedParts)-1):
                    validSD = getValidShifting(problem.width, problem.height, newCombinedParts[j], newCombinedParts[j+1])
                    #print(f"Currentpos: {newCombinedParts[j]}, nextPos: {newCombinedParts[j+1]}, validDirections: {validSD}")
                    newTuple = (newCombinedParts[j+1], validSD[random.randint(0, len(validSD)-1)])
                    tmp.append(newTuple)
                
                #Fuze everything back together
                new_genes = part_one + tmp + part_two
                X_mut[i][0] = new_genes
                
        return np.array(X_mut)
    
    def find_midpoint(self, y1, x1, y2, x2):
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        return (int(mid_y), int(mid_x))
    
    def sample_point_within_radius(self, mid_y, mid_x, radius, problem, individual):
        while True:
            rand_y = mid_y + random.randint(-radius, radius)
            rand_x = mid_x + random.randint(-radius, radius)
            
            if 0 <= rand_y < problem.height and 0 <= rand_x < problem.width:
                if (rand_y, rand_x) not in individual:
                    return (rand_y, rand_x)
    

    def manhattan_distance(self, point1, point2):
        """Calculate the Manhattan distance between two points."""
        #print("Manhatten")
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


    def greedy_path_find(self, start, end, problem):
        """Find a path from start to end using greedy best-first search."""
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
                if 0 <= move[0] < problem.height and 0 <= move[1] < problem.width
            ]
            
            if not valid_moves:
                break  # No valid moves available

            # Choose the move that minimizes the heuristic (Manhattan distance to the end)
            next_pos = min(valid_moves, key=lambda move: self.manhattan_distance(move, end))
            
            # Add the chosen move to the path
            path.append(next_pos)
            current_pos = next_pos

        # Remove the last point from the path if it's the same as 'end'
        if path[-1] == end:
            path.pop()

        return path