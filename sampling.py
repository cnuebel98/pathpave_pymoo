from pymoo.core.sampling import Sampling
import numpy as np

class RandomSampling(Sampling):
    def __init__(self, width, height, start, end):
        super().__init__()
        self.width = width
        self.height = height
        self.start = start
        self.end = end

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples , 1), None, dtype=object)
        for i in range(n_samples):
            path = self._random_path()
            X[i, 0] =path
        return X

    def _random_path(self):
        path = [self.start]
        current_pos = self.start

        while current_pos != self.end:
            row, col = current_pos
            # Generate all possible moves (up, down, left, right)
            possible_moves = [
                (row-1, col), (row+1, col), 
                (row, col-1), (row, col+1)
            ]
            # Filter out invalid moves
            valid_moves = [
                move for move in possible_moves 
                if 0 <= move[0] < self.height and 0 <= move[1] < self.width and move != current_pos
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
                next_pos = valid_moves[np.random.randint(len(valid_moves))]
                path.append(next_pos)
                current_pos = next_pos
            else:
                break  # If no valid moves are left, break the loop
        #print("Random Path for initial Pop: " + str(path))
        return path

class RandomDirectionSampling(Sampling):
    def __init__(self, width, height, start, end):
        super().__init__()
        self.width = width
        self.height = height
        self.start = start
        self.end = end

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples , 1), None, dtype=object)
        for i in range(n_samples):
            path = self._random_path()
            X[i, 0] =path
        return X

    def _random_path(self):
        possibleShiftingDirections = [(self.start[0] + 1, self.start[1] + 0),
                                      (self.start[0] + 0, self.start[1] + 1),
                                      (self.start[0] - 1, self.start[1] + 0),
                                      (self.start[0] + 0, self.start[1] - 1)]
        # Filter out invalid shifting directions
        # Also remove current pos since shifting weights through yourself is not possible
        possibleShiftingDirections = [
            direction for direction in possibleShiftingDirections 
            if 0 <= direction[0] < self.height and 0 <= direction[1] < self.width
        ]
        start_x, start_y = self.start
        startTuple = (self.start, possibleShiftingDirections[np.random.randint(len(possibleShiftingDirections))])
        path = [startTuple]
        current_pos = self.start

        while current_pos != self.end:
            row, col = current_pos
            # Generate all possible moves (up, down, left, right)
            possible_moves = [
                (row-1, col), (row+1, col), 
                (row, col-1), (row, col+1)
            ]
            # Filter out invalid moves
            valid_moves = [
                move for move in possible_moves 
                if 0 <= move[0] < self.height and 0 <= move[1] < self.width and move != current_pos
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
                next_pos = valid_moves[np.random.randint(len(valid_moves))]
                possibleShiftingDirections = [(next_pos[0] + 1, next_pos[1] + 0),
                                              (next_pos[0] + 0, next_pos[1] + 1),
                                              (next_pos[0] - 1, next_pos[1] + 0),
                                              (next_pos[0] + 0, next_pos[1] - 1)]
                # Filter out invalid shifting directions
                # Also remove current pos since shifting weights through yourself is not possible
                possibleShiftingDirections = [
                    direction for direction in possibleShiftingDirections 
                    if 0 <= direction[0] < self.height and 0 <= direction[1] < self.width and direction != current_pos
                ]
                next_tuple = (next_pos, possibleShiftingDirections[np.random.randint(len(possibleShiftingDirections))])
                path.append(next_tuple)
                current_pos = next_pos
            else:
                break  # If no valid moves are left, break the loop
        #print("Random Path for initial Pop: " + str(path))
        return path