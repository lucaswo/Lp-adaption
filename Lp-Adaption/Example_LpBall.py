import numpy as np
import json
import Vol_lp
import LpBallSampling
import LpAdaption

class LpBallExample():
    def __init__(self,dim:int,pnorm:int, optsFile:str = '../Inputs/opts_example.json'):
        with open(optsFile) as file:
           self.optsDict = json.load(fp=file)
        self.dim =dim
        self.pn = pnorm

    def lp_adaption(self):
        pn2 = 1
        r2 = 1
        mu2 = np.zeros(shape=(self.dim,1))
        q2 = np.diag(np.sqrt(np.logspace([0,3,3])))
        q2 = q2/(np.linalg.det(q2)**(1/self.dim))

        #true volume
        vol_t = Vol_lp.vol_lp(self.dim,r2,pn2)
        numRep = 2
        outCell = []

        for i in range(1,numRep):
            print('_____________ Repitition:',i,'______________')
            tmp = LpBallSampling.LpBall(self.dim,pn2).samplefromball(number=1)
            xstart = mu2 + r2*(q2@tmp)

            out = LpAdaption.LpAdaption(self.oracle(),xstart,inopts=self.optsDict)
            outCell.append(out)

        volVec = np.empty(shape=(numRep,len(self.optsDict['PVec'])))
        #TODO plotting
    def oracle(self):
        return 1



