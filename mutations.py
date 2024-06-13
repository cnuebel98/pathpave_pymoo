from pymoo.core.mutation import Mutation
import copy
import random
import numpy as np

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
                #print()
                #print("Individual to Mutate: " + str(X_mut[i][0]))
                #print("len of indi to mutate: " + str(len(X_mut[i][0])))

                # Find a random point in the individual to cut it into two
                individual_length = len(X_mut[i][0])
                
                start_point_for_swap = random.randint(1, individual_length-1)
                
                part_one = X_mut[i][0][:start_point_for_swap]
                part_two = X_mut[i][0][start_point_for_swap:]

                #print("X: " + str(X_mut[i][0]))
                #print("part 1: " + str(part_one))
                #print("part 2: " + str(part_two))
                
                # To not change the Individual too much, the longer 
                # sequence of the Gene stays and the shorter one is 
                # getting replaced so we either need to find a new
                # way to the start cell or the end cell, depending on the cut

                if len(part_one) >= len(part_two):
                    # keep part one and find new random way to end cell
                    new_part = self._generate_random_path(problem, X_mut[i][0][start_point_for_swap], problem.end)
                    if new_part and part_one[-1] == new_part[0]:
                        new_part = new_part[1:]
                    new_genes = part_one + new_part
                elif len(part_two) > len(part_one):
                    # keep part two and find new random way to start cell
                    new_part = self._generate_random_path(problem, X_mut[i][0][start_point_for_swap], problem.start)

                    # here we have to reverse the newly generated list before appending it to the part_two because we are finding a new way to the start
                    new_part_mirrored = new_part[::-1]
                    
                    # Then we can add this to the new gene set
                    if new_part_mirrored and new_part_mirrored[-1] == part_two[0]:
                        new_part_mirrored = new_part_mirrored[:-1]
                    new_genes = new_part_mirrored + part_two
                    
                #print("New_genes: " + str(new_genes))
                # Concatenate the parts to form the new individual

                #print("Mutated Gene Set: " + str(new_genes))
                X_mut[i][0] = new_genes
        return np.array(X_mut)
    
    def _generate_random_path(self, problem, start, end):
        
        # Your custom logic to generate a random path through the grid
        # This could be similar to what was implemented in the GridWorldProblem
        # Length between 20 and 100
        path = [start]
        current_pos = start

        #print("Curr_Pos at beginnning of new path: " + str(current_pos))
        #print("Final Position to reach: " + str(end))
        
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
        
        #print("New Part: " + str(path))
        return path
    