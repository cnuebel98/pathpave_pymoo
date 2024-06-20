from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np

class EliminateDuplicates(ElementwiseDuplicateElimination):
    
    
    def is_equal(self, a, b):
        """Compare two paths""" 
        
        # If the paath of two individuals is not the same
        if a.X[0] != b.X[0]:
            return False
        #print("a: " + str(a.X[0]))
        #print("b: " + str(b.X[0]))
        # If the shape matches, compare the decision variables
        return np.allclose(a.X[0], b.X[0], atol=0)