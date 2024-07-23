from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np

class EliminateDuplicates(ElementwiseDuplicateElimination):
    
    
    def is_equal(self, a, b):
        """Compare two paths""" 
        
        # If the path of two individuals is not the same
        print("")
        if a.X[0] != b.X[0]:
            return False
        #print("a: " + str(a.X[0]))
        #print("b: " + str(b.X[0]))
        # If the shape matches, compare the decision variables
        return np.allclose(a.X[0], b.X[0], atol=0)

class EliminateDirectionalDuplicates(ElementwiseDuplicateElimination):
    
    
    def is_equal(self, a, b):
        """Compare two paths""" 
        
        # If the path of two individuals is not the same
        if len(a.X[0]) > len(b.X[0]):
            length = len(b.X[0])
        else:
            length = len(a.X[0])
        
        for i in range(length):
            if a.X[0][i][0] != b.X[0][i][0]:
                #print("FOUND DUPLICATE")
                return False
        #print("a: " + str(a.X[0]))
        #print("b: " + str(b.X[0]))
        # If the shape matches, compare the decision variables
        #return np.allclose(a.X[0][i][0], b.X[0][i][0], atol=0)
        return True