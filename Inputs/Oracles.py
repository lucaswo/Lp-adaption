import numpy as np

class Oracle():
    def oracle(self,vector):
        return np.random.random_integers(0,1,size=(1,1))
