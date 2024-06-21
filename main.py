import matplotlib.pyplot as plt
import numpy as np 

from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2

from sampling import RandomSampling
from mutations import CopyMutation, ChangePartsMutation
from crossovers import CopyCrossover, CrossingCrossover, onePointCrossover
from problem import GridWorldProblem
from duplicate_handling import EliminateDuplicates
from pathRepair import pathRepair

from pymoo.util.ref_dirs import get_reference_directions
from obstacles import Obstacles
# Define parameters
width = 50
height = 50
seed = 42

# Set start and end points
start = (height - 1, width // 2)  # Bottom middle
end = (0, width // 2) 

# Set Crossover and Mutation Probabilities
mutation_rate = 0.1
prob_crossover = 0.8

# Create an instance of the Obstacles class
obstacles = Obstacles(width, height, seed)

# Generate the desired obstacles on the map
#obstacle_map = obstacles.create_random_obstacles()
obstacle_map = obstacles.create_obstacles_bubble_in_middle()
#obstacle_map = obstacles.create_sinusoidal_obstacles()
#obstacle_map = obstacles.create_gradient_obstacles()

# Define the problem
problem = GridWorldProblem(width, height, obstacle_map, start, end)

# Usage:
pop_size = 10
sampling = RandomSampling(width, height, start, end)
#crossover = CrossingCrossover(prob_crossover=prob_crossover)
#crossover = CopyCrossover()
crossover = onePointCrossover(prob_crossover, (width, height))
mutation = ChangePartsMutation(mutation_rate=mutation_rate)
eliminate_duplicates = EliminateDuplicates()
repair = pathRepair()
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=5)

# Initialize the NSGA2 algorithm
#Use the following line for Random Selection. Otherwise its binary Tournament Selection 
#selection=RandomSelection(), 

nsga2 = NSGA2(pop_size=pop_size, 
              sampling=sampling, 
              crossover=crossover, 
              mutation=mutation, 
              repair=repair,
              eliminate_duplicates=eliminate_duplicates)

nsga3 = NSGA3(ref_dirs=ref_dirs,
              pop_size=pop_size, 
              sampling=sampling, 
              crossover=crossover, 
              mutation=mutation, 
              repair=repair,
              eliminate_duplicates=eliminate_duplicates)

spea2 = SPEA2(pop_size=pop_size, 
              sampling=sampling, 
              crossover=crossover, 
              mutation=mutation, 
              repair=repair,
              eliminate_duplicates=eliminate_duplicates)

moead = MOEAD(ref_dirs=ref_dirs,
              sampling=sampling, 
              #eliminate_duplicates=eliminate_duplicates,
              crossover=crossover, 
              mutation=mutation,
              repair=repair)

# Run optimization
res = minimize(problem
               #,nsga2
               ,nsga3
               #,spea2
               #,moead # moead doesn't want to take duplicate elimination
               ,('n_eval', 100)
               ,seed=seed
               ,verbose=True)


#print("res.pop: " + str(res.F))

# Extract the Pareto front data
pareto_front = res.F

#print(pareto_front[:, 0])

# Plot the Pareto front
plt.figure(figsize=(10, 8))
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], label='Pareto Front', color='b')

# Customize the plot
plt.xlabel('Steps Taken')
plt.ylabel('Weight Shifted')
plt.title('Pareto Front from NSGA-II')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
# Extract the paths from res.X
paths = res.X.squeeze().tolist()

# Print the paths
#print("Paths:")
#for path in paths:
#    print(path)

# Create a plot for the final grid with paths
fig, ax = plt.subplots(figsize=(13, 8))

# Display the obstacle weights in the grid
#for i in range(height):
  #  for j in range(width):
    #    ax.text(j, i, f'{obstacles[i, j]:.2f}', va='center', ha='center', fontsize=12)

# Plot the grid
ax.imshow(obstacle_map, cmap='Greys', interpolation='nearest')

# Mark the start and end points
ax.plot(start[1], start[0], 'go', markersize=10, label='Start')  # Start point
ax.plot(end[1], end[0], 'ro', markersize=10, label='End')        # End point

# Plot the paths of the final population
if len(paths[0]) != 2:
    for path in paths:
        path_y, path_x = zip(*path)
        ax.plot(path_x, path_y, marker='o')
elif len(paths[0]) == 2:
    path_y, path_x = zip(*paths)
    ax.plot(path_x, path_y, marker='o')

# Set the ticks and labels
ax.set_xticks(np.arange(width))
ax.set_yticks(np.arange(height))
ax.set_xticklabels(np.arange(width))
ax.set_yticklabels(np.arange(height))

plt.title("GridWorld with Obstacle Weights and Final Paths")
plt.show()