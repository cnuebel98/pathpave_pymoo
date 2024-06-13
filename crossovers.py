from pymoo.core.crossover import Crossover
import copy

class CopyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)
    
    def _do(self, problem, X, **kwargs):
        #print("CopyCrossover")
        return copy.deepcopy(X)  # Return deep copies of the parent


class CrossingCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)
    
    def _do(self, problem, X, **kwargs):
        X_crossed = copy.deepcopy(X)
        print(X_crossed)
        print("len von pop: " + str(len(X_crossed)))
        print(type(X_crossed))

        for individual_list in X_crossed:
            for individual in individual_list:
                gene_sequence = individual[0]
                for gene in gene_sequence:
                    #print("hi " + str(gene))
                    ...
            ## list of list with first Individal
            #print("hi1" + str(individual[0]))
            # list with first Individal
            #print("hi2" + str(individual[0][0]))
            # first coordinate of Individual
            #print("hi3" + str(individual[0][0][0]))
            # X coordinate of first coordinate of Individual
            #print("hi4: " + str(individual[0][0][0][0]))
            
        return X_crossed