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
            p['valP'] = self.opts.hitP_adapt['PVec']
        else:
            p['valP'] = self.opts.valP
        #initialize values for the parameters from the options
        p['nOut'] = self.opts.nOut
        p['mueff'] = 1
        p['N_mu'] = self.opts.N_mu
        p['N_C'] = self.opts.N_C
        p['ccov1'] = self.opts.ccov1
        p['ccovmu'] = self.opts.ccovmu
        p['popSize'] = self.opts.popSize
        p['cp'] = self.opts.cp
        p['l_expected'] = np.sqrt(self.N)
        p['windowSize'] = np.ceil(self.opts.windowSizeEval)
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
            test=0
            #TODO. Implement adaptable hitting probabilitie
        else:
            numLast = self.opts.numLast
            #Check, that numLast is smaller than MaxEval
            if numLast > self.opts.maxEval:
                numLast = max((self.opts.maxEval - 1e3*self.N),self.opts.maxEval * 0.7)
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
        if xstart_out != 1:
            ValueError('x_start needs to be a feasable point')

        lastxAcc = self.xstart
        #Number of function evaluations
        counteval= 1
        if p['hitP_adapt']:
            vcounteval = 1
            vcountgeneration = 1

            if self.opts.para_hitP_adapt.fixedSchedule:
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
        





lp = LpAdaption(xstart=[1,1])
lp.lpAdaption()
