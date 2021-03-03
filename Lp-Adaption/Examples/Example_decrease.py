import json
import sys
sys.path.append(".")
sys.path.append("..")

import LpAdaption
import numpy as np




class ExampleDecrease():

    def __init__(self, optsFile: str = '../Inputs/example_decrease.json'):
        with open(optsFile) as file:
            self.optsDict = json.load(fp=file)

    def lp_adaption(self):
        xstart = np.array([[-1], [-1]])
        l = LpAdaption.LpAdaption(oracle=self.oracle_heavyTailedStar, xstart=xstart, inopts=self.optsDict)
        out = l.lpAdaption()
        # figure
        # TODO Plot

    def oracle_heavyTailedStar(self, x):
        # cast x to an np array fpr the case it's a list
        n, number = x.shape

        if n != 2:
            ValueError('Wrong dimension of input')

        a = 1.1
        b = 1.35
        mu1 = np.array([b, 0])
        q1 = np.array([[a, 0], [0, 1]])
        r1 = 0.9

        mu2 = np.array([-b, 0])
        q2 = q1

        mu3 = np.array([0, b])
        q3 = np.array([[1, 0], [0, a]])

        mu4 = [0, -b]
        q4 = q3

        # if only one point is given
        if number == 1:
            f0 = (np.sum(np.power(np.abs(0.5 * x), 2)) ** 0.5) <= 1
            f1 = (np.sum(np.abs(np.linalg.inv(q1) / r1 * (x - mu1))**2) ** 0.5) <= 1
            f2 = (np.sum(np.abs(np.linalg.inv(q2) / r1 * (x - mu2))**2) ** 0.5) <= 1
            f3 = (np.sum(np.abs(np.linalg.inv(q3) / r1 * (x - mu3))**2) ** 0.5) <= 1
            f4 = (np.sum(np.abs(np.linalg.inv(q4) / r1 * (x - mu4))**2) ** 0.5) <= 1
        else:
            x1 = x.T
            f0 = (np.sum(np.power(np.abs(0.5 * x1), 2), axis=1) ** 0.5).T <= 1

            mu = mu1.T
            b = (x1 - np.tile(mu, (1, number)))
            xtest = np.linalg.inv(q1) / r1 * b
            f1 = np.sum(np.abs(xtest) ** 2) ** 0.5 <= 1

            mu = mu2.T
            b = (x1 - np.tile(mu, (1, number)))
            xtest = np.linalg.inv(q2) / r1 * b
            f2 = np.sum(np.abs(xtest) ** 2) ** 0.5 <= 1

            mu = mu3.T
            b = (x1 - np.tile(mu, (1, number)))
            xtest = np.linalg.inv(q3) / r1 * b
            f3 = np.sum(np.abs(xtest) ** 2) ** 0.5 <= 1

            mu = mu4.T
            b = (x1 - np.tile(mu, (1, number)))
            xtest = np.linalg.inv(q4) / r1 * b
            f4 = np.sum(np.abs(xtest) ** 2) ** 0.5 <= 1

        return [f0 and not f1 and not f2 and not f3 and not f4]


l = ExampleDecrease()
l.lp_adaption()
