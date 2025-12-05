##############
# Run with:
# python3 main.py --map=0 --w=15 --h=15 --algo=0 --cross=0 --mut=0 --pop=50 --neval=1000 --shift=0 --seed=42
##############

import matplotlib.pyplot as plt
import numpy as np 
import argparse
import time

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
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.ctaea import CTAEA

from sampling import RandomSampling
from mutations import CopyMutation, ChangePartsMutation, RectangleMutation, RadiusSamplingMutation
from crossovers import CopyCrossover, CrossingCrossover, OnePointCrossover, TwoPointCrossover
from problem import GridWorldProblem
from duplicate_handling import EliminateDuplicates
from repairs import ErrorRepair, PathRepair
from callback import Callback

from pymoo.util.ref_dirs import get_reference_directions
from obstacles import Obstacles

from logger import Logger
#Create logger
log = Logger()

# Define parameters
width = 51
height = 51
seed = 42


def show_map_with_path():
    # Create a plot for the final grid with paths
    fig, ax = plt.subplots(figsize=(7, 7))

    # Display the obstacle weights in the grid
    #for i in range(height):
    #    for j in range(width):
    #        ax.text(j, i, f'{obstacles[i, j]:.2f}', va='center', ha='center', fontsize=12)
    start = (50,25)
    end = (0,25)
    paths = [
        
    #[(50, 25), (49, 25), (48, 25), (48, 24), (48, 23), (48, 22), (48, 21), (48, 20), (48, 19), (48, 18), (48, 17), (47, 17), (47, 16), (46, 16), (45, 16), (44, 16), (44, 15), (44, 14), (44, 13), (43, 13), (42, 13), (41, 13), (40, 13), (39, 13), (38, 13), (37, 13), (36, 13), (35, 13), (34, 13), (33, 13), (32, 13), (31, 13), (30, 13), (30, 14), (30, 15), (30, 16), (29, 16), (29, 17), (28, 17), (28, 18), (28, 19), (28, 20), (27, 20), (27, 21), (26, 21), (26, 22), (25, 22), (25, 23), (25, 24), (25, 25), (24, 25), (23, 25), (22, 25), (21, 25), (20, 25), (19, 25), (18, 25), (17, 25), (16, 25), (15, 25), (14, 25), (13, 25), (12, 25), (11, 25), (10, 25), (9, 25), (8, 25), (7, 25), (6, 25), (5, 25), (4, 25), (3, 25), (2, 25), (1, 25), (0, 25)]
    #[(50, 25), (49, 25), (48, 25), (48, 24), (48, 23), (48, 22), (48, 21), (48, 20), (48, 19), (48, 18), (48, 17), (47, 17), (47, 16), (46, 16), (45, 16), (44, 16), (44, 15), (44, 14), (44, 13), (44, 12), (43, 12), (43, 11), (42, 11), (41, 11), (40, 11), (39, 11), (38, 11), (37, 11), (36, 11), (35, 11), (34, 11), (33, 11), (32, 11), (32, 12), (32, 13), (31, 13), (30, 13), (30, 14), (30, 15), (30, 16), (29, 16), (29, 17), (28, 17), (28, 18), (28, 19), (28, 20), (27, 20), (27, 21), (26, 21), (26, 22), (26, 23), (25, 23), (25, 24), (25, 25), (25, 26), (24, 26), (24, 27), (24, 28), (24, 29), (23, 29), (22, 29), (22, 30), (22, 31), (21, 31), (20, 31), (20, 32), (20, 33), (20, 34), (19, 34), (19, 35), (19, 36), (19, 37), (18, 37), (17, 37), (16, 37), (15, 37), (14, 37), (13, 37), (12, 37), (11, 37), (10, 37), (9, 37), (8, 37), (8, 36), (7, 36), (6, 36), (5, 36), (5, 35), (5, 34), (5, 33), (4, 33), (3, 33), (3, 32), (3, 31), (2, 31), (1, 31), (1, 30), (1, 29), (1, 28), (1, 27), (0, 27), (0, 26), (0, 25)]
    [(50, 25), (49, 25), (48, 25), (47, 25), (46, 25), (45, 25), (44, 25), (43, 25), (42, 25), (41, 25), (40, 25),(40, 24), (40, 23), (40, 22), (40, 21), (40, 20), (40, 19), (40, 18), (40, 17), (40, 16), (40, 15),(39, 15), (38, 15), (37, 15), (36, 15), (35, 15), (34, 15), (33, 15), (32, 15), (31, 15), (30, 15),(30, 16), (30, 17), (30, 18), (30, 19), (30, 20), (30, 21), (30, 22), (30, 23), (30, 24), (30, 25),(30, 26), (30, 27), (30, 28), (30, 29), (30, 30), (30, 31), (30, 32), (30, 33), (30, 34), (30, 35),(29, 35), (28, 35), (27, 35), (26, 35), (25, 35), (24, 35), (23, 35), (22, 35), (21, 35), (20, 35),(19, 35), (18, 35), (17, 35), (16, 35), (15, 35), (14, 35), (13, 35), (12, 35), (11, 35), (10, 35),(10, 34), (10, 33), (10, 32), (10, 31), (10, 30), (10, 29), (10, 28), (10, 27), (10, 26), (10, 25),(10, 24), (10, 23), (10, 22), (10, 21), (10, 20),(9, 20), (8, 20), (7, 20), (6, 20), (5, 20),(5, 21), (5, 22), (5, 23), (5, 24), (5, 25),(4, 25), (3, 25), (2, 25), (1, 25), (0, 25)]
    ,[(50, 25), (50, 26), (50, 27), (50, 28), (50, 29), (50, 30),(49, 30), (48, 30), (47, 30), (46, 30), (45, 30), (44, 30), (43, 30), (42, 30), (41, 30), (40, 30), (39, 30), (38, 30), (37, 30), (36, 30), (35, 30),(35, 29), (35, 28), (35, 27), (35, 26), (35, 25), (35, 24), (35, 23), (35, 22), (35, 21), (35, 20),(34, 20), (33, 20), (32, 20), (31, 20), (30, 20), (29, 20), (28, 20), (27, 20), (26, 20), (25, 20), (24, 20), (23, 20), (22, 20), (21, 20), (20, 20),(20, 21), (20, 22), (20, 23), (20, 24), (20, 25), (20, 26), (20, 27), (20, 28), (20, 29), (20, 30), (20, 31), (20, 32), (20, 33), (20, 34), (20, 35), (20, 36), (20, 37), (20, 38), (20, 39), (20, 40),(19, 40), (18, 40), (17, 40), (16, 40), (15, 40), (14, 40), (13, 40), (12, 40), (11, 40), (10, 40), (9, 40), (8, 40), (7, 40), (6, 40), (5, 40),(5, 39), (5, 38), (5, 37), (5, 36), (5, 35), (5, 34), (5, 33), (5, 32), (5, 31), (5, 30),(4, 30), (3, 30), (2, 30), (1, 30), (0, 30),(0, 29), (0, 28), (0, 27), (0, 26), (0, 25)]
    ]


    
    obstacle_map = np.zeros((51,51))
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
    # Set x and y ticks in steps of 10
    ax.set_xticks(np.arange(0, width, 5))
    ax.set_yticks(np.arange(0, height, 5))

    # Set x and y tick labels in steps of 10
    ax.set_xticklabels(np.arange(0, width, 5))
    ax.set_yticklabels(np.arange(0, height, 5))

    #plt.title("Obstacle Environment")
    plt.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", help="Defines used map", type=int)
    parser.add_argument("--w", help="Defines width of the map", type=int)
    parser.add_argument("--h", help="Defines height of the map", type=int)
    parser.add_argument("--algo", help="Defines used algorithm", type=int)
    parser.add_argument("--cross", help="Defines used crossover", type=int)
    parser.add_argument("--mut", help="Defines used mutation", type=int)
    parser.add_argument("--pop", help="Defines population size", type=int)
    parser.add_argument("--neval", help="Number of function evaluations", type=int)
    parser.add_argument("--shift", help="Decides which method is used to shift the weight", type=int)
    parser.add_argument("--seed", help="Determines seed for semi random values", type=int)
    args = parser.parse_args()
    simulation(args.map, args.w, args.h, args.algo, args.cross, args.mut, args.pop, args.neval, args.shift, args.seed)
    #print(args.map)

def simulation(m, w, h, a, c, mut, p, n, sm, s):
    startingTime = time.time()
# Set shifiting method if defined
    if sm != None:
        shiftingMethod = sm
    else:
        shiftingMethod = 0

    if s != None:
        seed = s
    else:
        seed = 420

    # Set height and width if defined
    if w != None:
        width = w
    if h != None:
        height = h

    # Set start and end points
    start = (height - 1, width // 2)  # Bottom middle
    end = (0, width // 2)

    #start = (0, 0)
    #end = (width-1, height-1)

    # Set Crossover and Mutation Probabilities
    mutation_rate = 0.2
    prob_crossover = 0.8

    # Create an instance of the Obstacles class
    obstacles = Obstacles(width, height, seed)
    maps = [obstacles.create_sinusoidal_obstacles, 
            obstacles.create_gradient_obstacles,
            obstacles.create_radial_gradient_obstacles, 
            obstacles.create_meandering_river_obstacles,
            obstacles.create_meandering_river_obstacles_mirrored,
            obstacles.create_steep_gradient_obstacles]

    # Set map if defined
    if m != None:
        if (maps[m].__name__ != "create_random_walk_obstacles"):
            obstacle_map = maps[m]()
        else:
            obstacle_map = maps[m](num_walks=width*height)
    else:
        obstacle_map = obstacles.create_radial_gradient_obstacles()
 
    # Define the problem
    problem = GridWorldProblem(width, height, obstacle_map, start, end, shiftingMethod)

    # Usage:
    if p != None:
        pop_size = p
    else:
        pop_size = 50

    sampling = RandomSampling(width, height, start, end)

    crossovers = [CrossingCrossover(prob_crossover=prob_crossover), CopyCrossover(), OnePointCrossover(prob_crossover, (width, height)), TwoPointCrossover(prob_crossover, (width, height))]
    if c != None:
        crossover = crossovers[c]
    else:
        crossover = OnePointCrossover(prob_crossover, (width, height))

    mutations = [RadiusSamplingMutation(mutation_rate=mutation_rate, radius=int(0.2*height+0.2*width), problem=problem), RectangleMutation(mutation_rate=mutation_rate), ChangePartsMutation(mutation_rate)]

    if mut != None:
        mutation = mutations[mut]
    else:
        mutation = RadiusSamplingMutation(mutation_rate=mutation_rate, radius=int(0.2*height+0.2*width), problem=problem)

    eliminate_duplicates = EliminateDuplicates()
    repair = PathRepair()
    #repair = errorRepair()
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=10)

    # Initialize the NSGA2 algorithm
    #Use the following line for Random Selection in the algorithms. Otherwise its binary Tournament Selection 
    #selection=RandomSelection(), 
    algorithms = [NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=eliminate_duplicates),
                  NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair,eliminate_duplicates=eliminate_duplicates),
                  SPEA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair, eliminate_duplicates=eliminate_duplicates),
                  MOEAD(ref_dirs=ref_dirs, sampling=sampling, crossover=crossover, mutation=mutation, repair=repair),
                  RNSGA2(ref_points=ref_dirs, pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair = repair, eliminate_duplicates=eliminate_duplicates),
                  AGEMOEA(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair = repair, eliminate_duplicates=eliminate_duplicates),
                  AGEMOEA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation, repair = repair, eliminate_duplicates=eliminate_duplicates), # probleme
                  DNSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation,repair=repair, eliminate_duplicates=eliminate_duplicates),
                  SMSEMOA(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation,repair=repair, eliminate_duplicates=eliminate_duplicates),
                  CTAEA(ref_dirs=ref_dirs, sampling=sampling, crossover=crossover, mutation=mutation, eliminate_duplicates=eliminate_duplicates)]

    if a != None:
        algorithm = algorithms[a]
    else:
        algorithm = algorithms[0]

    if n != None:
        n_eval = n
    else:
        n_eval = 1000

    # Create a callback object
    callback = Callback()

    # Run optimization
    res = minimize(problem
                   ,algorithm
                   ,('n_eval', n_eval)
                   ,seed=seed
                   ,verbose=True
                   ,callback=callback)

    totalTime = time.time() - startingTime
    
    # Extract the Pareto front data
    pareto_front = res.F
    #LOGGING
    log.createLogFile(obstacles, width, height, algorithm, crossover, mutation, pop_size, n_eval, sampling, repair, shiftingMethod, seed, totalTime)

    #print(pareto_front[:, 0])

    # Extract the Pareto optimal paths and fitness values
    po_fitness_values_per_gen = callback.data["po_f_values"]
    po_paths_per_gen = callback.data["po_paths"]
    all_fitness_values_per_gen = callback.data["all_f_values"]
    all_paths_per_gen = callback.data["all_paths"]
    
    #print(len(all_fitness_values_per_gen)) # = 200 for 200 Generations
    #print(len(all_fitness_values_per_gen[0])) # = 50 for 50 individuals
    #print(all_paths_per_gen)
    #print(po_fitness_values_per_gen)
    #print(po_paths_per_gen)

    # Plot the Pareto front
    # plt.figure(figsize=(10, 8))
    
    # plt.scatter(po_fitness_values_per_gen[0][:, 0], po_fitness_values_per_gen[0][:, 1], label='First Pareto Front', color='b')
    # plt.scatter(po_fitness_values_per_gen[-1][:, 0], po_fitness_values_per_gen[-1][:, 1], label='Last Pareto Front', color='r')
    
    #plt.scatter(all_fitness_values_per_gen[:, 0], all_fitness_values_per_gen[:, 1], label='Pareto Front', color='b')

    # Customize the plot
    # plt.xlabel('Steps Taken')
    # plt.ylabel('Total Weight Shifted')
    # plt.title('Pareto Front')
    # plt.legend()
    # plt.grid(True)
    # Show the plot
    #plt.show()
    # Save plot
    #plt.savefig(log.logPath+"/paretoPlot")
    # Extract the paths from res.X
    #paths = res.X.squeeze().tolist()
    
    # This would draw all paths of the final population
    paths = po_paths_per_gen[-1].squeeze().tolist()
    #print(paths)
    #print(len(paths))
    #print(len(paths[0]))
    #paths = all_paths_per_gen[-1].squeeze().tolist()
    
    # Print the paths
    #print("Paths:")
    #for path in paths:
    #    print(path)

    # plot paths on map:
    #show_map_with_path()
    
    #plt.savefig(log.logPath+"/mapPlot")
    log.log(paths, pareto_front[:, 0], pareto_front[:, 1])

if __name__ == "__main__":
    main()

