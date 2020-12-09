import numpy as np
from typing import List,Dict
import re
from DefaultOptions import DefaultOptions
import json

class LpAdaption:

    def __init__(self,oracle:str, xstart:List,inopts:str=''):
        '''
        :param oracle: Python File in Inputs directory
         containing oracle class with oracle function
        :param xstart: feasable point to start with
        :param inopts: input options
        '''
        try:
            oracle_cleaned = re.sub('.py','',oracle)
            from Inputs import oracle_cleaned
            self.oracle = oracle_cleaned.Oracle()
        except:
            ValueError('There has to be an Oracle defined in the Inputs folder\ '
                       'Or at least it has to be a String name of the python file without the .py'
                       'Example: OracleExample')
        # define dimension N trough starting point
        self.N = len(xstart)
        #load default options
        self.opts = DefaultOptions(self.N)
        self.xstart = xstart
        if inopts:
            try:
                #swap default paramters in inopts
                with open(inopts, 'r') as inopts_file:
                    inopts_dict = json.load(inopts_file)
                    self.opts.adaptOption(inopts_dict)
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

        #__________SetUp of algorithmic parameters______________
        maxMeanSize = self.opts.maxMeanSize
        if self.opts.hitP_adapt_cond:
            valP = self.opts.hitP_adapt['PVec']
        else:
            valP = self.opts.valP

        nOut = self.opts.nOut
        mueff = 1
        N_mu = eval(self.opts.N_mu)
        N_C = self.opts.N_C
        ccov1 = eval(self.opts.ccov1)
        ccovmu = eval(self.opts.ccovmu)
        popSize = eval(self.opts.popSize)
        cp = self.opts.cp
        l_expected = np.sqrt(self.N)
        windowSize = np.ceil(eval(self.opts.windowSizeEval))
        beta = eval(self.opts.beta)
        ss = eval(self.opts.ss)
        sf = eval(self.opts.sf)
        r = self.opts.r
        rMax = self.opts.maxR
        rMin = self.opts.minR
        condMax = self.opts.maxCond
        pn = self.opts.pn
        p_empAll = 0
        p_empWindow = 0

        if self.opts.hitP_adapt_cond:
            test=0
            #TODO. Implement adaptable hitting probabilitie
        else:
            numLast = eval(self.opts.numLast)
            #Check, that numLast is smaller than MaxEval
            if numLast > self.opts.maxEval:
                numLast = max((self.opts.maxEval - 1e3*self.N),self.opts.maxEval * 0.7)
                Warning('Num Last is too big, it was changed to default %d.'%numLast)

        averageCoNum = self.opts.averageCovNum

        Q = np.array(self.opts.initQ)
        C = np.array(self.opts.initC)
        #C consistent calculated from Q for compairison with specified C
        C_calc= r**2*(Q*Q)
        np.testing.assert_equal(C,C_calc,err_msg='Initialized C Matrix is not consistent with Matrix Q.\n'
                                                     'Your Q yields to the following C:'%C_calc)
        #check if C is positiv semidefinit
        [Bo,tmp] = np.linalg.eig(C)
        






lp = LpAdaption(oracle='OracleExample',xstart=[1,1])
lp.lpAdaption()
