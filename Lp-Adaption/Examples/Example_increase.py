import json
import sys
sys.path.append(".")
sys.path.append("..")

import LpAdaption
import numpy as np



class ExampleIncrease():

    def __init__(self, optsFile: str = '../Inputs/example_increase.json'):
        with open(optsFile) as file:
            self.optsDict = json.load(fp=file)

    def lp_adaption(self):
        xstart = np.array([[-1], [0]])
        l = LpAdaption.LpAdaption(oracle=self.oracle_pacman, xstart=xstart, inopts=self.optsDict)
        out = l.lpAdaption()


    def oracle_pacman(self, x):
        # cast x to an np array fpr the case it's a list
        n, number = x.shape

        if n != 2:
            ValueError('Wrong dimension of input')

        mu = np.array([0.3, 0])
        Q = np.array([[2.5, 0],[0, 1]])
        r = 0.3

        if number==1:
            f1 = (np.sum(np.abs(x)**2)**0.5)<=1
            f2 = (np.sum(np.abs(np.linalg.inv(Q)/r*(x-mu).T)**2)**0.5)<=1
        else:
            x1 = x.T
            f1 = ((np.sum(np.abs(x1)**2,axis=1)**0.5)<=1).T

            mu1 = mu.T
            b = x1 - np.tile(mu1,(1,number))
            xtest = np.linalg.inv(Q)/r*(b)

            f2 = ((np.sum(np.abs(xtest)**2,axis=1)**0.5)<=1).T

        return [f1 and not f2]

l = ExampleIncrease()
l.lp_adaption()
