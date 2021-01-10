import numpy as np
import sklearn
from scipy.stats import gengamma
import random


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
        psi = gengamma.rvs(a=(1/self.pnorm), c=1, size=(number,self.dim))
        psi = [x ** (1 / self.pnorm) for x in psi]
        psi = np.array(psi)

        # generate number x dim random signs for the samples psi to multiply with
        signs = np.random.randint(2, size=(number, self.dim))
        signs[signs == 0] = -1
        X = np.multiply(psi,signs)
        # calc z = w^/dim, where w is random variable in [0,1]
        # copy z to dim dimensional array
        z = np.random.rand(number,1) ** (1 / self.dim)
        i = np.abs(X)**self.pnorm
        l = np.array(np.sum(np.abs(X) ** self.pnorm,axis=1)**(1. / self.pnorm))
        m = np.transpose(np.tile(l,(self.dim,1)))
        y = np.tile(z,(1,self.dim))*(X/m)
        return y


