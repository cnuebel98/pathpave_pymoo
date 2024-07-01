import matplotlib.pyplot as plt
import numpy as np 
import argparse

from pymoo.optimize import minimize
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.dnsga2 import DNSGA2

from sampling import RandomSampling
from mutations import CopyMutation, ChangePartsMutation, RectangleMutation, RadiusSamplingMutation
from crossovers import CopyCrossover, CrossingCrossover, OnePointCrossover
from problem import GridWorldProblem
from duplicate_handling import EliminateDuplicates
from repairs import ErrorRepair, PathRepair

from pymoo.util.ref_dirs import get_reference_directions
from obstacles import Obstacles

from logger import Logger
#Create logger
log = Logger()

# Define parameters
width = 50
height = 50
seed = 42

parser = argparse.ArgumentParser()
parser.add_argument("--map", help="Defines used map", type=int)
parser.add_argument("--w", help="Defines width of the map", type=int)
parser.add_argument("--h", help="Defines height of the map", type=int)
parser.add_argument("--algo", help="Defines used algorithm", type=int)
parser.add_argument("--cross", help="Defines used crossover", type=int)
parser.add_argument("--mut", help="Defines used mutation", type=int)
parser.add_argument("--pop", help="Defines population size", type=int)
parser.add_argument("--neval", help="Number of function evaluations", type=int)
args = parser.parse_args()
#print(args.map)
# Set start and end points
start = (height - 1, width // 2)  # Bottom middle
end = (0, width // 2)

# Set height and width if defined
if args.w != None:
    width = args.w
if args.h != None:
    height = args.h

#start = (0, 0)
#end = (width-1, height-1)

# Set Crossover and Mutation Probabilities
mutation_rate = 0.1
prob_crossover = 0.8

# Create an instance of the Obstacles class
obstacles = Obstacles(width, height, seed)
maps = [obstacles.create_random_obstacles(), obstacles.create_obstacles_bubble_in_middle(), obstacles.create_sinusoidal_obstacles(), obstacles.create_gradient_obstacles(),
        obstacles.create_radial_gradient_obstacles(), obstacles.create_perlin_noise_obstacles(), obstacles.create_random_walk_obstacles(num_walks=width*height), obstacles.create_maze_obstacles()]

# Set map if defined
if args.map != None:
    obstacle_map = maps[args.map]
else:
    obstacle_map = obstacles.create_gradient_obstacles()
 
# Define the problem
problem = GridWorldProblem(width, height, obstacle_map, start, end)

# Usage:
if args.pop != None:
    pop_size = args.pop
else:
    pop_size = 50

sampling = RandomSampling(width, height, start, end)

crossovers = [CrossingCrossover(prob_crossover=prob_crossover), CopyCrossover(), OnePointCrossover(prob_crossover, (width, height))]
if args.cross != None:
    crossover = crossovers[args.cross]
else:
    crossover = OnePointCrossover(prob_crossover, (width, height))

mutations = [RadiusSamplingMutation(mutation_rate=mutation_rate, radius=int(0.1*height+0.1*width), problem=problem), RectangleMutation(mutation_rate=mutation_rate), ChangePartsMutation(mutation_rate)]

if args.mut != None:
    mutation = mutations[args.mut]
else:
    mutation = RadiusSamplingMutation(mutation_rate=mutation_rate, radius=int(0.1*height+0.1*width), problem=problem)

eliminate_duplicates = EliminateDuplicates()
repair = PathRepair()
#repair = errorRepair()
ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=5)

# Initialize the NSGA2 algorithm
#Use the following line for Random Selection in the algorithms. Otherwise its binary Tournament Selection 
#selection=RandomSelection(), 
algorithms = [NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation,repair=repair, eliminate_duplicates=eliminate_duplicates),
              NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair,eliminate_duplicates=eliminate_duplicates),
              SPEA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair, eliminate_duplicates=eliminate_duplicates),
              MOEAD(ref_dirs=ref_dirs, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair),
              RNSGA2(ref_points=ref_dirs, pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair = repair, eliminate_duplicates=eliminate_duplicates),
              AGEMOEA(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair = repair, eliminate_duplicates=eliminate_duplicates),
              AGEMOEA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair = repair, eliminate_duplicates=eliminate_duplicates),
              DNSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation,repair=repair, eliminate_duplicates=eliminate_duplicates)]

if args.algo != None:
    algorithm = algorithms[args.algo]
else:
    algorithm = algorithms[0]

if args.neval != None:
    n_eval = args.neval
else:
    n_eval = 1000
# Run optimization
res = minimize(problem
               ,algorithm
               ,('n_eval', n_eval)
               ,seed=seed
               ,verbose=True)


#print("res.pop: " + str(res.F))

# Extract the Pareto front data
pareto_front = res.F
#LOGGING
log.createLogFile(obstacles, width, height, algorithm, crossover, mutation, pop_size, n_eval, sampling, repair)

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
#plt.show()
# Save plot
plt.savefig(log.logPath+"/paretoPlot")
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
#    for j in range(width):
#        ax.text(j, i, f'{obstacles[i, j]:.2f}', va='center', ha='center', fontsize=12)

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

plt.title("Obstacle Environemnt")
# plt.show()
plt.savefig(log.logPath+"/mapPlot")
log.log(paths, pareto_front[:, 0], pareto_front[:, 1])