import numpy as np
import matplotlib.pyplot as plt


class PlotData():
    def __init__(self, N: int,name:str):
        self.n = N
        self.lasteigvals = np.ones(shape=(N, 1),dtype=int)
        self.last_ind = 0
        self.lastsave = 1
        self.name = name
        self.fig, self.axs = plt.subplots(2, 2)
        self.fig.figsize = (10,7.5)
        self.fig.suptitle(name)

    def plot(self, cntVec, countgeneration, saveIndGeneration, muVec, rVec, VerboseModuloGen, N, eigVals, r,
             P_empVecAll, P_empVecWindow, P_empAll, P_empWindow):
        currsaves = np.arange(self.lastsave-1, saveIndGeneration)
        currindices = cntVec[currsaves]
        currind = int(currindices[-1])


        self.axs[0,0].plot(currindices, muVec[currsaves, :])
        self.axs[0,0].set_title('mean')
        self.axs[0, 0].grid(True)

        line, = self.axs[0,1].plot(currindices,rVec[currsaves,:])
        self.axs[0, 1].set_yscale('log')
        self.axs[0, 1].grid(True)
        self.axs[0,1].set_title('current step size')

        self.axs[1,0].plot(currindices,P_empVecWindow[currsaves],c = 'red')
        self.axs[1,0].plot(currindices,P_empVecAll[currsaves],c='blue')
        self.axs[1,0].set_title('Current acceptace Probability all: window(red)')
        self.axs[1,0].grid(True)

        self.axs[1, 1].grid(True)
        self.axs[1,1].set_yscale('log')


        self.axs[1, 1].plot([self.last_ind,currind], [(1 / np.sort(self.lasteigvals)).reshape(N,), 1 / np.sort(eigVals)])

        if self.name == 'current Repition':
            self.fig.show()

        self.lastsave = saveIndGeneration
        self.last_ind = int(currindices[-1])
        self.lasteigvals = eigVals


