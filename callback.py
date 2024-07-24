import matplotlib.pyplot as plt
import numpy as np
#import pymoo.core.individual.Individual

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.problems import get_problem
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["objectiveValues"] = []
        self.data["paths"] = []
        self.data["optObjectiveValues"] = []
        self.data["optPaths"] = []

    def notify(self, algorithm):
        self.data["objectiveValues"].append(algorithm.pop.get("F"))
        self.data["paths"].append(algorithm.pop.get("_X"))
        self.data["optObjectiveValues"].append(algorithm.opt.get("F"))
        self.data["optPaths"].append(algorithm.opt.get("_X"))