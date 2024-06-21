import matplotlib.pyplot as plt
import numpy as np 

from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.algorithms.moo.nsga2 import NSGA2

from sampling import RandomSampling
from mutations import CopyMutation, ChangePartsMutation
from crossovers import CopyCrossover, CrossingCrossover, onePointCrossover
from problem import GridWorldProblem
from duplicate_handling import EliminateDuplicates
from pathRepair import pathRepair

# Define parameters
width = 20
height = 20
seed = 42

# Set start and end points
start = (height - 1, 0)
end = (0, width - 1)

# Set Crossover and Mutation Probabilities
mutation_rate = 0.1
prob_crossover = 0.1

# Generate obstacles
np.random.seed(seed)
obstacles = np.round(np.random.rand(height, width), 2)

# Define the problem
problem = GridWorldProblem(width, height, obstacles, start, end)

# Usage:
pop_size = 10
sampling = RandomSampling(width, height, start, end)
#crossover = CrossingCrossover(prob_crossover=prob_crossover)
#crossover = CopyCrossover()
crossover = onePointCrossover(prob_crossover, (width, height))
mutation = ChangePartsMutation(mutation_rate=mutation_rate)
eliminate_duplicates = EliminateDuplicates()
repair = pathRepair()


# Initialize the algorithm
algorithm = NSGA2(pop_size=pop_size, 
                  sampling=sampling, 
                  crossover=crossover, 
                  mutation=mutation,
                  repair = repair,
                  # Use the following line for Random Selection. Otherwise its binary Tournament Selection 
                  #selection=RandomSelection(), 
                  eliminate_duplicates=eliminate_duplicates)

# Run optimization
res = minimize(problem,
               algorithm,
               ('n_eval', 200),
               seed=seed,
               verbose=True)


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
print("Paths:")
for path in paths:
    print(path)

# Create a plot for the final grid with paths
fig, ax = plt.subplots(figsize=(13, 8))

# Display the obstacle weights in the grid
for i in range(height):
    for j in range(width):
        ax.text(j, i, f'{obstacles[i, j]:.2f}', va='center', ha='center', fontsize=12)

# Plot the grid
ax.imshow(obstacles, cmap='Blues', interpolation='nearest')

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