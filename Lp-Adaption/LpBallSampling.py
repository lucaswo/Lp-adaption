import numpy as np
import sklearn
from scipy.stats import gamma
import random
import matplotlib.pyplot as plt


class LpBall:
    """This class  intitializes an Lp-Ball within a
    given Dimension N and a pnorm (default:2)"""

    def __init__(self, dim: int, pnorm: int = 2):
        self.dim = dim
        self.pnorm = pnorm

        if dim is None:
            raise ValueError('Dimension is not defined!')

    def samplefromball(self, number: int = 1):
        '''
        Samples number of real value vectors for Lp-Adaption
        :param number: Number of samples (vectors), default :1
        :return: uniformly distributed real random vectors from the Lp-Ball
        '''
        # sample from gamma generalized distribution
        psi = np.random.default_rng().standard_gamma((1/self.pnorm),size=(number,self.dim))
        psi = psi**(1/self.pnorm)

        # generate number x dim random signs for the samples psi to multiply with
        signs = np.random.randint(2, size=(number, self.dim))
        signs[signs == 0] = -1
        X = psi*signs
        # calc z = w^/dim, where w is random variable in [0,1]
        # copy z to dim dimensional array
        z = np.random.rand(number,1) ** (1 / self.dim)
        y = np.tile(z,(1,self.dim))*(X/np.tile(np.sum(np.abs(X)**self.pnorm,axis=1)**(1/self.pnorm),(1,self.dim)).reshape(self.dim,number).T)
        return y




