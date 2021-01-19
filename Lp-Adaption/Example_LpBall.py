import numpy as np
import json
import Vol_lp
import LpBallSampling
import LpAdaption


class LpBallExample():

    def __init__(self, dim: int, pnorm: int, optsFile: str = '../Inputs/opts_example.json'):
        with open(optsFile) as file:
            self.optsDict = json.load(fp=file)
        self.dim = dim
        self.pn = pnorm

    def lp_adaption(self):
        pn2 = 1
        r2 = 1
        mu2 = np.zeros(shape=(self.dim, 1))
        q2 = np.diag(np.sqrt(np.logspace([0, 3, 3])))
        q2 = q2 / (np.linalg.det(q2) ** (1 / self.dim))

        # true volume
        vol_t = Vol_lp.vol_lp(self.dim, r2, pn2)
        numRep = 2
        outCell = []

        for i in range(1, numRep):
            print('_____________ Repitition:', i, '______________')
            tmp = LpBallSampling.LpBall(self.dim, pn2).samplefromball(number=1)
            xstart = mu2 + r2 * (q2 @ tmp)

            out = LpAdaption.LpAdaption(self.oracle, xstart, inopts=self.optsDict)
            outCell.append(out)

        volvec = np.empty(shape=(numRep, len(self.optsDict['PVec'])))
        # TODO plotting

    def oracle(self, x, np, r, mu, q):
        """
        :param x: candidate solutions -> matric with column vetors as candidate solutions
        :param np: p-norm of lp ball
        :param r: radius of lp ball
        :param mu: center of lp ball
        :param q: deformation matrix
        :return: 1 if x is inside LP Ball, 0 else
                When x was a matrix, the output is a vector of 0s and 1s
        """
        if len(x[:, 0]) == 1:
            xtest = np.linalg.inv(q) / r * (x - mu)
            if np > 100:
                if max(np.abs(xtest)) <= 1:
                    return 1
                else:
                    return 0
            else:
                if sum(np.abs(xtest) ** np) ** (1 / np) <= 1:
                    return 1
                else:
                    return 0
        else:
            number = len(x[:, 0])
            b = (x - np.tile(mu, (1, number)))
            xtest = np.linalg.inv(q) / b * r

            if np > 100:
                return (np.max(np.abs(xtest), axis=1) <= 1).astype('int')
            else:
                return np.transpose(np.sum(np.abs(xtest) ** np) ** (1 / np) <= 1).astype('int')
