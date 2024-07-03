import main as m
import os
import random
from multiprocessing import Process, Pool, freeze_support, get_context

def main():
    maps = [0,1,2,3,4,5,6,7]
    width = 100
    height = 100
    algorithms = [0,1,2,3,4,5,6,7]
    crossovers = [0,1,2]
    mutations = [0,1,2]
    pop = 100
    n_eval = 100000
    #n_eval = 1000
    shiftingMethods = [0,1,2]
    seeds = random.sample(range(1, 1000000), 31)
    combinations = getCombinations(maps, width, height, algorithms, crossovers, mutations, pop, n_eval, shiftingMethods, seeds)
    callMultiprocessing(combinations)

def callMultiprocessing(combinations: list):
    with get_context("fork").Pool(48) as pool:
        result = pool.map(multiProcessSimulations, combinations)
        pool.close

def getCombinations(maps, width, height, algorithms, crossovers, mutations, pop, n_eval, shiftingMethods, seeds) -> list:
    combinations = []
    for map in maps:
        for algorithm in algorithms:
            for crossover in crossovers:
                for mutation in mutations:
                    for shiftingMethod in shiftingMethods:
                        for seed in seeds:
                            combinations.append([map, width, height, algorithm, crossover, mutation, pop, n_eval, shiftingMethod, seed])
    return combinations

def multiProcessSimulations(c: list):
    m.simulation(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9])

if __name__ == "__main__":
    freeze_support()
    main()