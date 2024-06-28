import os
import shutil
import matplotlib.pyplot as plt
import json
import numpy as np

class logger():
    basePath = "./log"
    def __init__(self) -> None:
        """Init method that creates log directory if it does not exist."""
        if not os.path.exists("./log"):
            os.mkdir("./log")
    
    def createLogFile(self, map, width, height, algorithm, crossover, mutation, population, n_eval):
        """Creates a logfile for the path."""
        self.logName = f"{map.name}_{width}_{height}_{algorithm.__class__.__name__}_{crossover.__class__.__name__}_{mutation.__class__.__name__}_{population}_{n_eval}"
        self.logPath = self.basePath + "/" + self.logName
        # If a log for this already exists delete it
        if os.path.exists(self.logPath):
            shutil.rmtree(self.logPath)
        os.mkdir(self.logPath)
        f = open(self.logPath+"/"+self.logName+".json", 'w')
        f.close()

        # Set class variables
        self.map = map.name
        self.width = width
        self.height = height
        self.algorithm = algorithm.__class__.__name__
        self.crossover = crossover.__class__.__name__
        self.mutation = mutation.__class__.__name__
        self.population = population
        self.n_eval = n_eval
    
    def log(self, paths):
        """Creates a log object and writes it to the json file."""
        log_obj = {
            "map": self.map,
            "width": self.width,
            "height": self.height,
            "algorithm": self.algorithm,
            "crossover": self.crossover,
            "mutation": self.mutation,
            "population": self.population,
            "n_eval": self.population,
            "paths": paths,
        }
        #TODO: This converts array of tuples into array of arrays -> Dont let it do this
        with open(self.logPath+"/"+self.logName+".json", 'w') as f:
            json.dump(log_obj, f)
            f.close()