import numpy as np
import random

from pymoo.core.problem import Problem

class GridWorldProblem(Problem):
    def __init__(self, width, height, obstacles, start, end):
        super().__init__(n_var=1, n_obj=2, n_constr=0)
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start = start
        self.end = end

    def shift_obstacle_weight_randomly(self, current_grid, current_cell, next_cell): 
        # Takes the current Grid, and the field that has to be cleared as well as the
        #print()
        #print("Curr Grid:")
        #print(current_grid)
        #print("Current Cell: " + str(current_cell))
        #print("Next Cell: " + str(next_cell))
        #print("Weight to move: " + str(current_grid[next_cell[0], next_cell[1]]))

        updated_grid = current_grid
        
        # Generate all possible moves (up, down, left, right)
        row, col = next_cell

        possible_neighbors = [
            (row-1, col), (row+1, col), 
            (row, col-1), (row, col+1)
        ]
        # Filter out invalid moves
        valid_shifts = [
            shift for shift in possible_neighbors 
            if 0 <= shift[0] < self.height and 0 <= shift[1] < self.width and shift != current_cell
        ]
        #print("Valid_shifts: " + str(valid_shifts))
        
        # Select one of the valid shifts randomly
        chosen_shift = random.choice(valid_shifts)
        
        #print("Chosen shift: " + str(chosen_shift))
        
        # Get the weight from next_cell
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # delete weight from next_cell
        updated_grid[next_cell[0], next_cell[1]] = 0

        # shit weight randomly to possible neighbor
        updated_grid[chosen_shift[0], chosen_shift[1]] += weight_to_move
        #print(updated_grid)
        # return updated grid
        return updated_grid

    def least_resistance_shift(self, current_grid, current_cell, next_cell): 
        # Takes the current Grid, and the field that has to be cleared as well as the
        #print()
        #print("Curr Grid:")
        #print(current_grid)
        #print("Current Cell: " + str(current_cell))
        #print("Next Cell: " + str(next_cell))
        #print("Weight to move: " + str(current_grid[next_cell[0], next_cell[1]]))

        updated_grid = current_grid
        
        # Get the weight from next_cell
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]
        #print("Weight To Move: " + str(weight_to_move))
        # Generate all possible moves (up, down, left, right)
        row, col = next_cell

        possible_neighbors = [
            (row-1, col), (row+1, col), 
            (row, col-1), (row, col+1)
        ]

        valid_shifts = []
        # Loop through each possible neighboring cell
        for shift in possible_neighbors:
            row, col = shift  # Extract the row and column from the current neighboring cell

            # Check if the neighboring cell is within the grid boundaries
            if 0 <= row < self.height and 0 <= col < self.width:
                # Ensure the neighboring cell is not the same as the current cell
                if shift != current_cell:
                    #print("Shift: " + str(shift))
                    valid_shifts.append(shift)  # Add the valid neighboring cell to the list
        
        #print(valid_shifts)
        #print(len(valid_shifts))
        if len(valid_shifts) == 1:
            # if there is only one valid shift, we have to take it
            chosen_shift = valid_shifts[0]

        else:
            chosen_shift = valid_shifts[0]
            
            for i in valid_shifts:
                
                if updated_grid[i[0], i[1]] < updated_grid[chosen_shift[0], chosen_shift[1]]:
                    chosen_shift = i

                #print()
                #print("Valid_shifts: " + str(i))
                #print("X valid shift: " + str(i[0]))
                #print("Y valid shift: " + str(i[1]))
                #print("Weight of the Cell to Shift to: " + str(updated_grid[i[0], i[1]]))
                #print()

            # Select one of the valid shifts randomly
            #print("chosen shift: " + str(updated_grid[chosen_shift[0], chosen_shift[1]]))
        #print(updated_grid)
        
        #print("Chosen shift: " + str(chosen_shift))
        
        # Get the weight from next_cell
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # delete weight from next_cell
        updated_grid[next_cell[0], next_cell[1]] = 0

        # shit weight randomly to possible neighbor
        updated_grid[chosen_shift[0], chosen_shift[1]] += weight_to_move
        #print(updated_grid)
        
        return updated_grid

    def _evaluate(self, X, out, *args, **kwargs):
        # Input: whole population X
        # Output: steps taken and summed obstacle weights collected
        steps_all, obstacles_all = [], []
        current_obstacles = self.obstacles
        for x in X:
            current_obstacles = self.obstacles.copy()
            steps, obstacle_weights_summed = 0, 0
            for i in range(len(x)):
                for j in range(len(x[i])):
                    # For eachs tep taken by an individual, we need to alter the 
                    # obstacles in the grid
                    # print("Current Individual: " + str(x[i]))
                    
                    if j != 0:
                        current_cell = x[i][j-1]
                        next_cell = x[i][j]
                    else: 
                        current_cell = x[i][j]
                        next_cell = x[i][j]

                    obstacle_weights_summed += current_obstacles[x[i][j][0], x[i][j][1]]
                    #new_obstacles = self.shift_obstacle_weight_randomly(current_obstacles, current_cell, next_cell)
                    new_obstacles = self.least_resistance_shift(current_obstacles, current_cell, next_cell)
                    
                    current_obstacles = new_obstacles

            steps = len(x[0])
            steps_all.append(steps)
            obstacles_all.append(obstacle_weights_summed)

        out["F"] = np.column_stack([steps_all, obstacles_all])