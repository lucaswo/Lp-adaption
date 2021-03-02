import numpy as np
import matplotlib.pyplot as plt


class PlotData():
    def __init__(self, N: int):
        self.n = N
        self.lasteigvals = np.ones(shape=(N, 1))
        self.last_ind = 0
        self.lastsave = 1

        self.fig, self.axs = plt.subplots(2, 2)
    def plot(self, cntVec, countgeneration, saveIndGeneration, muVec, rVec, VerboseModuloGen, N, eigVals, r,
             P_empVecAll, P_empVecWindow, P_empAll, P_empWindow):
        currsaves = np.arange(self.lastsave, saveIndGeneration)
        currindices = cntVec[currsaves]
        currind = currindices[-1]


        self.axs[0,0].plot(currindices, muVec[currsaves, :])
        self.axs[0,0].set_title('mean')

        line, = self.axs[0,1].plot(currindices,rVec[currsaves,:])
        self.axs[0, 1].set_yscale('log')
        self.axs[0, 1].grid(True)
        self.axs[0,1].set_title('current step size')

        self.axs[1,0].plot(currindices,P_empVecWindow[currsaves],c = 'red')
        self.axs[1,0].plot(currindices,P_empVecAll[currsaves],c='blue')
        self.axs[1,0].set_title('Current acceptace Probability all: window(red)')

        eigValsD = np.diag(eigVals)
        self.axs[1, 1].grid(True)
        self.axs[1,1].set_yscale('log')
        self.axs[1, 1].set_xscale('log')
        if self.n ==2:
            self.axs[1,1].plot([self.last_ind,currind],[1/np.sort(self.lasteigvals),1/np.sort(eigValsD)].T)
        else:
            self.axs[1, 1].plot([self.last_ind, currind], [(1 / np.sort(self.lasteigvals)).reshape(3,), 1 / np.sort(eigVals)])


        self.fig.show()
        plt.show()

        self.lastsave = saveIndGeneration
        self.last_ind = currindices[-1]
        self.lasteigvals = eigVals


