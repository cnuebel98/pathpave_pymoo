import random
import numpy as np
from scipy.ndimage import gaussian_filter
from noise import pnoise2

class Obstacles:
    '''Various obstacle maps can be created here'''
    def __init__(self, width, height, seed):
        self.width = width
        self.height = height
        self.seed = seed
        self.name = None

    def create_random_obstacles(self):
        self.name = "randomObstacles"
        # Generate obstacles
        np.random.seed(self.seed)
        # randomly
        random_obstacles = np.round(np.random.rand(self.height, self.width), 2)
        return random_obstacles
    
    def create_obstacles_bubble_in_middle(self):
        self.name = "bubbleInMiddle"
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
        self.name = "gradientObstacles"
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
        self.name = "sinusoidalObstacles"
        # x*pi -> with that x, the number of peaks and valleys can be altered
        x = np.linspace(0, 5 * np.pi, self.width)
        y = np.linspace(0, 5 * np.pi, self.height)
        x, y = np.meshgrid(x, y)
        # combining sine and cosine functions
        sinusoidal_obstacles = np.sin(x) * np.cos(y)
        # Normalize to 0-1 range
        sinusoidal_obstacles = (sinusoidal_obstacles - sinusoidal_obstacles.min()) / (sinusoidal_obstacles.max() - sinusoidal_obstacles.min())
        return sinusoidal_obstacles
    
    def create_radial_gradient_obstacles(self):
        self.name = "radialGradientObstacles"
        # Center of the grid
        center_x, center_y = self.width // 2, self.height // 2

        # Create a distance matrix from the center
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        distances = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)

        # Normalize distances to range [0, 1]
        normalized_distances = distances / np.max(distances)

        # Invert the distances to create a gradient from center to edges
        radial_gradient = 1 - normalized_distances

        return radial_gradient
    
    def create_random_walk_obstacles(self, num_walks, walk_length=1000, p=0.5):
        self.name = "randomWalkObstacles"
        # Initialize obstacle map
        obstacles = np.zeros((self.height, self.width))
        for _ in range(num_walks):
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            for _ in range(walk_length):
                obstacles[y, x] = 1
                dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])
                x = (x + dx) % self.width
                y = (y + dy) % self.height
                if np.random.rand() > p:
                    break
        return obstacles
    
    def create_perlin_noise_obstacles(self, scale=10.0, octaves=6, persistence=0.5, lacunarity=2.0):
        self.name = "perlinNoiseObstacles"
        # Generate Perlin noise using numpy and adjust parameters as needed
        noise = np.zeros((self.height, self.width))
        for i in range(self.height):
            for j in range(self.width):
                noise[i, j] = pnoise2(i / scale,
                                        j / scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=self.width,
                                        repeaty=self.height,
                                        base=self.seed)
        # Normalize to 0-1 range
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        return noise



    def create_maze_obstacles(self):
        self.name = "mazeObstacles"
        # Initialize maze with all walls
        maze = np.ones((self.height, self.width), dtype=int)

        # Start DFS from a random odd position
        start_x = np.random.randint(1, self.width // 2) * 2 + 1
        start_y = np.random.randint(1, self.height // 2) * 2 + 1
        self._recursive_dfs(maze, start_x, start_y)

        return maze

    def _recursive_dfs(self, maze, x, y):
        # Set current cell as walkable
        maze[y, x] = 0

        # Define movement directions (right, left, down, up)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Randomize directions
        np.random.shuffle(directions)

        # Visit neighboring cells in randomized order
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy  # Neighbor coordinates

            # Check if neighbor is within bounds and is a wall
            if 1 <= nx < self.width and 1 <= ny < self.height and maze[ny, nx] == 1:
                # Remove wall between current cell and neighbor
                maze[y + dy, x + dx] = 0
                self._recursive_dfs(maze, nx, ny)