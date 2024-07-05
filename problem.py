import numpy as np
import random
import copy
from pymoo.core.problem import Problem
from typing import Tuple

class GridWorldProblem(Problem):
    def __init__(self, width, height, obstacles, start, end, shiftingMethod):
        super().__init__(n_var=1, n_obj=2, n_constr=0)
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start = start
        self.end = end
        self.shiftingMethod = shiftingMethod

    def shift_obstacle_weight_randomly(self, current_grid, current_cell, next_cell): 
        # Takes the current Grid, and the field that has to be cleared as well as the

        updated_grid = copy.deepcopy(current_grid)
        
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

        # Select one of the valid shifts randomly
        chosen_shift = random.choice(valid_shifts)
        
        # Get the weight from next_cell
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # delete weight from next_cell
        updated_grid[next_cell[0], next_cell[1]] = 0

        # shit weight randomly to possible neighbor
        updated_grid[chosen_shift[0], chosen_shift[1]] += weight_to_move
        # return updated grid
        return updated_grid

    def least_resistance_shift(self, current_grid, current_cell, next_cell): 
        # Makes a copy to update the Obstacle grid
        updated_grid = copy.deepcopy(current_grid)
        
        # Get the weight from next_cell
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]
        
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
                    valid_shifts.append(shift)  # Add the valid neighboring cell to the list

        if len(valid_shifts) == 1:
            # if there is only one valid shift, we have to take it
            chosen_shift = valid_shifts[0]

        else:
            chosen_shift = valid_shifts[0]
            
            for i in valid_shifts:
                
                if updated_grid[i[0], i[1]] < updated_grid[chosen_shift[0], chosen_shift[1]]:
                    chosen_shift = i
        
        # Get the weight from next_cell
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # delete weight from next_cell
        updated_grid[next_cell[0], next_cell[1]] = 0

        # shit weight to the neighbor cell with the least amount of obstacle weight on it
        updated_grid[chosen_shift[0], chosen_shift[1]] += weight_to_move

        return updated_grid

    def split_weights_in_half_shift(self, current_grid: np.ndarray, current_cell: Tuple[int, int], next_cell: Tuple[int, int]) -> np.ndarray:
        updated_grid = copy.deepcopy(current_grid)
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # Determine the direction of movement
        direction = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])

        # Handle the special case where direction is (0, 0)
        if direction == (0, 0):
            direction = (-1, 0)  # Assume upward direction
        
        # Determine left and right neighbors based on movement direction
        if direction == (0, 1):  # Moving right (x increases)
            left_neighbor = (next_cell[0], next_cell[1] - 1)
            right_neighbor = (next_cell[0], next_cell[1] + 1)
        elif direction == (0, -1):  # Moving left (x decreases)
            left_neighbor = (next_cell[0], next_cell[1] + 1)
            right_neighbor = (next_cell[0], next_cell[1] - 1)
        elif direction == (1, 0):  # Moving down (y increases)
            left_neighbor = (next_cell[0] - 1, next_cell[1])
            right_neighbor = (next_cell[0] + 1, next_cell[1])
        elif direction == (-1, 0):  # Moving up (y decreases)
            left_neighbor = (next_cell[0] + 1, next_cell[1])
            right_neighbor = (next_cell[0] - 1, next_cell[1])
        else:
            raise ValueError("Invalid movement direction")

        # Check for wall conditions and split weight accordingly
        left_in_bounds = 0 <= left_neighbor[0] < self.height and 0 <= left_neighbor[1] < self.width
        right_in_bounds = 0 <= right_neighbor[0] < self.height and 0 <= right_neighbor[1] < self.width

        if left_in_bounds and right_in_bounds:
            half_weight = weight_to_move / 2
            updated_grid[left_neighbor[0], left_neighbor[1]] += half_weight
            updated_grid[right_neighbor[0], right_neighbor[1]] += half_weight
        elif left_in_bounds:
            updated_grid[left_neighbor[0], left_neighbor[1]] += weight_to_move
        elif right_in_bounds:
            updated_grid[right_neighbor[0], right_neighbor[1]] += weight_to_move
        else:
            raise ValueError("Both left and right neighbors are out of bounds")

        updated_grid[next_cell[0], next_cell[1]] = 0
        
        return updated_grid

    def split_weights_in_thirds_shift(self, current_grid: np.ndarray, current_cell: Tuple[int, int], next_cell: Tuple[int, int]) -> np.ndarray:
        updated_grid = copy.deepcopy(current_grid)
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # Determine the direction of movement
        direction = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])

        # Handle the special case where direction is (0, 0)
        if direction == (0, 0):
            direction = (-1, 0)  # Assume upward direction

        # Debugging output for direction and cells
        #print(f"Current cell: {current_cell}, Next cell: {next_cell}, Direction: {direction}")

        # Determine neighbors based on movement direction
        if direction == (0, 1):  # Moving right (x increases)
            left_neighbor = (next_cell[0] - 1, next_cell[1])
            right_neighbor = (next_cell[0] + 1, next_cell[1])
            forward_neighbor = (next_cell[0], next_cell[1] + 1)  # Moving right (y increases)
        elif direction == (0, -1):  # Moving left (x decreases)
            left_neighbor = (next_cell[0] + 1, next_cell[1])
            right_neighbor = (next_cell[0] - 1, next_cell[1])
            forward_neighbor = (next_cell[0], next_cell[1] - 1)  # Moving left (y decreases)
        elif direction == (1, 0):  # Moving down (y increases)
            left_neighbor = (next_cell[0], next_cell[1] + 1)
            right_neighbor = (next_cell[0], next_cell[1] - 1)
            forward_neighbor = (next_cell[0] + 1, next_cell[1])  # Moving down (x increases)
        elif direction == (-1, 0):  # Moving up (y decreases)
            left_neighbor = (next_cell[0], next_cell[1] - 1)
            right_neighbor = (next_cell[0], next_cell[1] + 1)
            forward_neighbor = (next_cell[0] - 1, next_cell[1])  # Moving up (x decreases)
        else:
            raise ValueError("Invalid movement direction")

        # Check for wall conditions and split weight accordingly
        left_in_bounds = 0 <= left_neighbor[0] < self.height and 0 <= left_neighbor[1] < self.width
        right_in_bounds = 0 <= right_neighbor[0] < self.height and 0 <= right_neighbor[1] < self.width
        forward_in_bounds = 0 <= forward_neighbor[0] < self.height and 0 <= forward_neighbor[1] < self.width

        #print(f"Left neighbor: {left_neighbor}, In bounds: {left_in_bounds}")
        #print(f"Right neighbor: {right_neighbor}, In bounds: {right_in_bounds}")
        #print(f"Forward neighbor: {forward_neighbor}, In bounds: {forward_in_bounds}")

        if left_in_bounds and right_in_bounds and forward_in_bounds:
            third_weight = weight_to_move / 3
            updated_grid[left_neighbor[0], left_neighbor[1]] += third_weight
            updated_grid[right_neighbor[0], right_neighbor[1]] += third_weight
            updated_grid[forward_neighbor[0], forward_neighbor[1]] += third_weight
        elif left_in_bounds and right_in_bounds:
            half_weight = weight_to_move / 2
            updated_grid[left_neighbor[0], left_neighbor[1]] += half_weight
            updated_grid[right_neighbor[0], right_neighbor[1]] += half_weight
        elif left_in_bounds:
            updated_grid[left_neighbor[0], left_neighbor[1]] += weight_to_move
        elif right_in_bounds:
            updated_grid[right_neighbor[0], right_neighbor[1]] += weight_to_move
        elif forward_in_bounds:
            updated_grid[forward_neighbor[0], forward_neighbor[1]] += weight_to_move
        else:
            raise ValueError("All neighbors are out of bounds")

        updated_grid[next_cell[0], next_cell[1]] = 0
        
        return updated_grid
    
    def directionalShift(self, current_grid: np.ndarray, current_cell: Tuple[int, int], next_cell: Tuple[int, int], shiftedCell: Tuple[int, int]) -> np.ndarray:
        updated_grid = copy.deepcopy(current_grid)
        weight_to_move = updated_grid[next_cell[0], next_cell[1]]

        # Determine the direction of movement
        direction = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])

        # Handle the special case where direction is (0, 0)
        if direction == (0, 0):
            direction = (-1, 0)  # Assume upward direction

        # Determine neighbors based on movement direction
        if direction == (0, 1):  # Moving right (x increases)
            left_neighbor = (next_cell[0] - 1, next_cell[1])
            right_neighbor = (next_cell[0] + 1, next_cell[1])
            forward_neighbor = (next_cell[0], next_cell[1] + 1)  # Moving right (y increases)
        elif direction == (0, -1):  # Moving left (x decreases)
            left_neighbor = (next_cell[0] + 1, next_cell[1])
            right_neighbor = (next_cell[0] - 1, next_cell[1])
            forward_neighbor = (next_cell[0], next_cell[1] - 1)  # Moving left (y decreases)
        elif direction == (1, 0):  # Moving down (y increases)
            left_neighbor = (next_cell[0], next_cell[1] + 1)
            right_neighbor = (next_cell[0], next_cell[1] - 1)
            forward_neighbor = (next_cell[0] + 1, next_cell[1])  # Moving down (x increases)
        elif direction == (-1, 0):  # Moving up (y decreases)
            left_neighbor = (next_cell[0], next_cell[1] - 1)
            right_neighbor = (next_cell[0], next_cell[1] + 1)
            forward_neighbor = (next_cell[0] - 1, next_cell[1])  # Moving up (x decreases)
        else:
            raise ValueError("Invalid movement direction")

        # Check for wall conditions and split weight accordingly
        left_in_bounds = 0 <= left_neighbor[0] < self.height and 0 <= left_neighbor[1] < self.width
        right_in_bounds = 0 <= right_neighbor[0] < self.height and 0 <= right_neighbor[1] < self.width
        forward_in_bounds = 0 <= forward_neighbor[0] < self.height and 0 <= forward_neighbor[1] < self.width

        #print(f"Left neighbor: {left_neighbor}, In bounds: {left_in_bounds}")
        #print(f"Right neighbor: {right_neighbor}, In bounds: {right_in_bounds}")
        #print(f"Forward neighbor: {forward_neighbor}, In bounds: {forward_in_bounds}")
        #print(f"left: {left_neighbor}, right: {right_neighbor}, forward: {forward_neighbor}, shiftedCell: {shiftedCell}, pos: {current_cell}")
        if left_in_bounds and shiftedCell==left_neighbor:
            updated_grid[left_neighbor[0], left_neighbor[1]] += weight_to_move
        elif right_in_bounds and shiftedCell==right_neighbor:
            updated_grid[right_neighbor[0], right_neighbor[1]] += weight_to_move
        elif forward_in_bounds and shiftedCell == forward_neighbor:
            updated_grid[forward_neighbor[0], forward_neighbor[1]] += weight_to_move
        elif current_cell == next_cell:
            updated_grid[shiftedCell[0], shiftedCell[1]] += weight_to_move
        else:
            raise ValueError(f"All neighbors are out of bounds or shifted cell is not a valid neighbor\nPosition: {current_cell}, nextCell: {next_cell}, shiftedCell: {shiftedCell}\n")

        updated_grid[next_cell[0], next_cell[1]] = 0
        return updated_grid

    def _evaluate(self, X, out, *args, **kwargs):
        # Input: whole population X
        # Output: steps taken and summed obstacle weights collected
        steps_all, obstacles_all = [], []
        current_obstacles = self.obstacles
        for x in X:
            current_obstacles = self.obstacles.copy()
            steps, obstacle_weights_summed = 0, 0
            #Big if since we can have individuals that have direction in encoding and some without
            if list(map(type, x[0][0])) == list(map(type, ((0,0),(0,0)))):
                for i in range(len(x)):
                    for j in range(len(x[i])):
                        shiftingDirection = x[i][j][1]
                        if j != 0:
                            current_cell = x[i][j-1][0]
                            next_cell = x[i][j][0]
                        else: 
                            current_cell = x[i][j][0]
                            next_cell = x[i][j][0]
                        obstacle_weights_summed += current_obstacles[x[i][j][0][0], x[i][j][0][1]]
                        new_obstacles = self.directionalShift(current_obstacles, current_cell, next_cell, shiftingDirection)
                        current_obstacles = new_obstacles
            else:
                for i in range(len(x)):
                    for j in range(len(x[i])):
                        # For eachs tep taken by an individual, we need to alter the 
                        # obstacles in the grid
                        #print("Current Individual: " + str(x[i]))

                        if j != 0:
                            current_cell = x[i][j-1]
                            next_cell = x[i][j]
                        else: 
                            current_cell = x[i][j]
                            next_cell = x[i][j]

                        obstacle_weights_summed += current_obstacles[x[i][j][0], x[i][j][1]]
                        if self.shiftingMethod == 0:
                            new_obstacles = self.shift_obstacle_weight_randomly(current_obstacles, current_cell, next_cell)
                        elif self.shiftingMethod == 1:
                            new_obstacles = self.least_resistance_shift(current_obstacles, current_cell, next_cell)
                        elif self.shiftingMethod == 2:
                            new_obstacles = self.split_weights_in_half_shift(current_obstacles, current_cell, next_cell)
                        elif self.shiftingMethod == 3:
                            new_obstacles = self.split_weights_in_thirds_shift(current_obstacles, current_cell, next_cell)

            steps = len(x[0])
            steps_all.append(float(steps))
            obstacles_all.append(obstacle_weights_summed)

        out["F"] = np.column_stack([steps_all, obstacles_all])