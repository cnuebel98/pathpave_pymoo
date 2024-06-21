import random
import numpy as np
from scipy.ndimage import gaussian_filter

class Obstacles:
    '''Various obstacle maps can be created here'''
    def __init__(self, width, height, seed):
        self.width = width
        self.height = height
        self.seed = seed

    def create_random_obstacles(self):
        # Generate obstacles
        np.random.seed(self.seed)
        # randomly
        random_obstacles = np.round(np.random.rand(self.height, self.width), 2)
        return random_obstacles
    
    def create_obstacles_bubble_in_middle(self):
        # Center of the grid
        center_x, center_y = self.width // 2, self.height // 2

        # Create a distance matrix from the center
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

        # Parameters for the normal distribution
        mean = 0  # center of the distribution
        std_dev = np.max(distances) / 2  # spread of the distribution

        # Generate obstacle weights using the normal distribution
        obstacle_weights = np.exp(-(distances - mean)**2 / (2 * std_dev**2))

        # Normalize the obstacle weights to the range [0, 1]
        obstacle_weights = (obstacle_weights - np.min(obstacle_weights)) / (np.max(obstacle_weights) - np.min(obstacle_weights))

        # Optionally, round the values to 2 decimal places
        obstacle_weights = np.round(obstacle_weights, 2)
        return obstacle_weights
    
    def create_gradient_obstacles(self):
        gradient = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                # Calculate the distance to the middle diagonal
                distance_to_diagonal = abs(i - j)
                max_distance = max(self.height, self.width) - 1
                normalized_distance = distance_to_diagonal / max_distance
                # The farther from the diagonal, the higher the value, reversed
                gradient[i, j] = 0.5 * (1 - normalized_distance)
                
                # Adjust for bottom right corner
                if i > j:
                    gradient[i, j] = 1 - gradient[i, j]
                else:
                    gradient[i, j] = gradient[i, j]
        
        return gradient

    def create_sinusoidal_obstacles(self):
        x = np.linspace(0, 4 * np.pi, self.width)
        y = np.linspace(0, 4 * np.pi, self.height)
        x, y = np.meshgrid(x, y)
        # Example of combining sine and cosine functions
        sinusoidal_obstacles = np.sin(x) * np.cos(y)
        # Normalize to 0-1 range
        sinusoidal_obstacles = (sinusoidal_obstacles - sinusoidal_obstacles.min()) / (sinusoidal_obstacles.max() - sinusoidal_obstacles.min())
        return sinusoidal_obstacles