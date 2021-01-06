import numpy as np
from typing import List,Dict
import re
from DefaultOptions import DefaultOptions
import json
from scipy.sparse.linalg import arpack
from Inputs import Oracles

class LpAdaption:

    def __init__(self, xstart:List,inopts=''):
        '''
        :param oracle: Python File in Inputs directory
         containing oracle class with oracle function
        :param xstart: feasable point to start with
        :param inopts: input options
        '''

        self.oracle = Oracles.Oracle()
        # define dimension N trough starting point
        self.N = len(xstart)
        #load default options
        self.opts = DefaultOptions(self.N)
        self.xstart = xstart
        if inopts:
            try:
                if type(inopts) == str:
                    #swap default paramters in inopts
                    with open(inopts, 'r') as inopts_file:
                        inopts_dict = json.load(inopts_file)
                        self.opts.adaptOption(inopts=inopts_dict)
                else:
                    self.opts.adaptOption(inopts=inopts)
            except:
                ValueError('Look up your json inopts file for mistakes')
        #rename option strings, so they can be found from this class
        for opt,value in self.opts.__dict__.items():
            if type(value) == str:
                setattr(self.opts,opt,re.sub('self.','self.opts.',value))
        self.isbSavingOn = self.opts.bSaving
        self.isPlottingOn = self.opts.plotting

    def lpAdaption(self):
        #Test if hitP_adapt was set and the condition not, already caught when merging defopts and inopts
        #Dictionaray of recent parameters
        p = {}
        #__________SetUp of algorithmic parameters______________
        maxMeanSize = self.opts.maxMeanSize
        if self.opts.hitP_adapt_cond:
            p['valP'] = self.opts.hitP_adapt['pVec']
        else:
            p['valP'] = self.opts.valP
        #initialize values for the parameters from the options
        p['nOut'] = self.opts.nOut
        p['maxEval'] = int(self.opts.maxEval)
        p['mueff'] = 1
        p['N_mu'] = self.opts.N_mu
        p['N_C'] = self.opts.N_C
        p['ccov1'] = self.opts.ccov1
        p['ccovmu'] = self.opts.ccovmu
        p['popSize'] = self.opts.popSize
        p['cp'] = self.opts.cp
        p['l_expected'] = np.sqrt(self.N)
        p['windowSize'] = np.ceil(self.opts.windowSizeEval).astype('int')
        p['beta'] = self.opts.beta
        p['ss'] = self.opts.ss
        p['sf'] = self.opts.sf
        p['r'] = self.opts.r
        p['rMax'] = self.opts.maxR
        p['rMin'] = self.opts.minR
        p['condMax'] = self.opts.maxCond
        p['pn'] = self.opts.pn
        p['p_empAll'] = 0
        p['p_empWindow'] = 0

        if self.opts.hitP_adapt_cond:
            p['pVec'] = self.opts.hitP_adapt['pVec']
            p['lpVec'] = len(p['pVec'])
            #TODO Implement adaptable hitting probability rest
        else:
            numLast = self.opts.numLast
            #Check, that numLast is smaller than MaxEval
            if numLast > p['maxEval']:
                numLast = max((p['maxEval'] - 1e3*self.N),p['maxEval'] * 0.7)
                Warning('Num Last is too big, it was changed to default %d.'%numLast)

        averageCoNum = self.opts.averageCovNum

        Q = np.array(self.opts.initQ)
        C = np.array(self.opts.initC)
        #C consistent calculated from Q for compairison with specified C
        C_calc= p['r']**2*(Q*Q)
        np.testing.assert_equal(C,C_calc,err_msg='Initialized C Matrix is not consistent with Matrix Q.\n'
                                                     'Your Q yields to the following C:'%C_calc)
        #check if C is positiv semidefinit
        #TODO: Maybe easier way
        [Bo,tmp] = np.linalg.eig(C)
        tol = 1e-8
        vals, vecs = arpack.eigsh(C, k=2, which='BE')
        if not np.all(vals > -tol):
            ValueError('Covariance Matrix need tobe positiv semidefinit!')
        elif np.size(C,1) != self.N:
            ValueError(' C has not the same size as your starting point xstart!')


        diagD = np.sqrt(np.diag(tmp))
        detdiagD = np.prod(diagD)
        diagD = diagD/(detdiagD**(1/self.N))
        Q = Bo*(np.tile(diagD,(self.N,1)))
        r = np.linalg.det(C)**(1/2*self.N)
        #matlab checks for the same size of C and xstart here again
        #Anyway, we not cause the check is already above and not in an if/else statement(cause initC and
        # Q are always provided through default Options)

        [j,eigVals] = np.linalg.eig(Q*np.transpose(Q))
        condC = np.linalg.cond(Q*np.transpose(Q))

        #___________Setup initial settings_____________
        #check if xstart is a feasable point
        xstart_out = self.oracle.oracle(self.xstart)
        len_x = len(xstart_out)
        #if dim of vector xstart and nOut Parameter differs, one might want to check the oracle
        if len_x != p['nOut']:
            UserWarning('Dimension of Oracle output differs from nOut option! nOut will be set so the len of xStart!')
            p['nOut'] = len_x
        if xstart_out[0] != 1:
            ValueError('x_start needs to be a feasable point')

        lastxAcc = self.xstart
        #Number of function evaluations
        counteval= 1
        if self.opts.hitP_adapt_cond:
            vcounteval = 1
            vcountgeneration = 1

            if self.opts.hitP_adapt['fixedSchedule']:
                cntsave_Part = 1
        else:
            #Number of evaluations after MaxEval - numLast evaluations
            countevalLast =0
            #Number of accepted points after MaxEval - numLast points
            lastNumAcc=0

        countgeneration = 1

        #Number of all accepted points equals one because we need to start with a feasible point!
        numAcc =1
        #Number of accepted points for specific hitP
        vNumAcc =1
        mu = self.xstart

        #___________Setup Output Parameters_____________
        if self.isbSavingOn:
            xRawDim = (np.ceil(p['maxEval']/self.opts.savingModulo).astype('int'),self.N)
            xRaw = np.empty(shape=xRawDim)
            xRaw[0,:] = self.xstart
            #save all accepted x to estimate the upper bound of the volume
            xAcc = np.empty((int(p['maxEval']),self.N))
            # counteval of all accepted x
            cntAcc = np.empty(shape=(int(p['maxEval']),1))

            cntAcc[0] = 1

            #oracle output is a vector out of 0s and 1s
            if p['nOut'] > 1:
                fxAcc = np.empty(shape=(int(p['maxEval']),p['nOut']-1))
                fxAcc[0,:] = xstart_out[1:]
            xAcc[0,:] = np.transpose(xstart_out)

            if self.opts.unfeasibleSave:
                xNotAcc = np.empty(shape=(int(p['maxEval']),p['nOut']-1))
                cntNotAcc = np.empty(shape=(p['maxEval'],1))
                if p['nOut']>1:
                    fxNotAcc = np.empty(shape=(int(p['maxEval']), p['nOut'] - 1))

            # Vector, if sample was accepted or not
            c_TVec = np.empty(shape=(np.ceil(p['maxEval']/self.opts.savingModulo).astype('int'),1))
            c_TVec[0] = xstart_out[0]

            # if output length of oracle is bigger than one, the following oracle values have to be saved
            if p['nOut']>1:
                fc_TVec = np.empty(shape=(np.ceil(p['maxEval'] / self.opts.savingModulo).astype('int'), p['nOut'] - 1))
                fc_TVec[0,:] = xstart_out[1:]

           # Vector of evaluation indices when everything is saved, first one is one, because xstart is feasable point
            #TODO: Ist Vector für die Gesamtentscheidungs-Speicherung?
            countVec = np.empty(shape=(np.ceil(p['maxEval'] / self.opts.savingModulo).astype('int'),1))
            countVec[0] = 1

            # TODO: in Matlab just a 1x1 cell for double value, init. here usefull?
            stopFlag = None

            #settings for CMA/GaA
            verboseModuloGen = np.ceil(self.opts.verboseModulo/p['popSize']).astype('int')
            savingModuloGen = np.ceil(self.opts.savingModulo/p['popSize']).astype('int')
            tmp_num = np.ceil(p['maxEval']/savingModuloGen).astype('int')
            if self.opts.hitP_adapt_cond:
                #how often hitting probability is changed
                cntAdapt=1
                # save different values for parameter when hitp changes
                n_MuVec = np.empty(shape = (p['lpVec'],1))
                n_MuVec[0] =p['N_mu']
                betaVec = np.empty(shape = (p['lpVec'],1))
                betaVec[0] = p['beta']
                ssVec = np.empty(shape = (p['lpVec'],1))
                ssVec[0] = p['ss']
                sfVec = np.empty(shape = (p['lpVec'],1))
                sfVec[0] = p['sf']
                iterVec = np.empty(shape = (p['lpVec'],1))
                windowSize = np.ceil(self.opts.windowSizeEval/p['popSize'])

            




lp = LpAdaption(xstart=[1,1])
lp.lpAdaption()
