import numpy as np
import json

class DefaultOptions:
    """
    This class handles all default parameters for the Lp_Adaption.
    All Parameters/options, which are not specified but necessary when starting the
    algorithm are taken from the default.
    All Options and their description can be additionally found in the README
    Options that are dependent on other options are written as string which is finally evaluated into the value
    #TODO: write all aptions with explanation to README
    """

    def __init__(self, N: int):
        # Dimension N has to be defined
        if N is None:
            raise ValueError("Dension has to be specified")
        else:
            self.N = N

        # maximum number of function evaluations
        self.maxEval = 1e4 * N

        # Pnorm
        self.pn = 2

        # ???
        self.oracleInopts = []

        # number of outputs from the oracle (first has to be 0/1), later mor values possible
        self.nOut = 1

        # Plot results: for irst tests off, TODO: Set default 'on' if plotting is available
        self.plotting = False  #

        # After wich iteration one want to have a message from Commandline.
        # MaxEval % verbosemodulo == 0
        self.verboseModulo = 1e3

        # Save after i-th iteration, maxeval % savingmodulo ==0
        self.savingModulo = 1e2

        # Save data to file on/off
        self.bSaving = True

        # Save Covaraiance Matrices (leads to huge files)
        self.bSaveCov = True

        # save all numLast r, mu, Q, P_emp
        self.lastSaveAll = False

        # Save unfeasable Solutions
        self.unfeasibleSave = False

        #how many of numLast elements are used to get average mu and r
        self.numLast = "max((self.maxEval - 1e3*self.N),self.maxEval * 0.7)"

        #how many covariances should be used to get average covariance
        self.averageCovNum = 100

        # options concerning algorithmic parameters:
        #---------------------------------------------
        #Hitting Probability
        self.valP = 1 / np.exp(1)

        #  upper bound on inverval's size over which averaging happens ???
        self.maxMeanSize = 2000

        # size of moving window to get empirical hitting probability (in number of evaluations)
        self.windowSizeEval = "min(110/self.valP,self.maxMeanSize)"

        # Initial Covariance
        self.r = 1

        #init Cholesky and Corvariance matrix
        self.initQ = np.eye(N).tolist()
        self.initC = np.eye(N).tolist()

        self.maxR = "np.inf"

        self.minR = 0

        self.maxCond = 1e20*N

        #Mean apdaption weight
        self.N_mu = np.exp(1)*N

        #Matrix adaption weight
        self.N_C = ((N+1.3)**2+1)/2

    #TODO find out why gamma population size can be also np.floor(2 / self.valP) --> not in default mentioned in paper
        self.popSize = "max(4 + np.floor(3 * np.log(self.N)), np.floor(2 / self.valP))"
        # Learning rate beta line 6 in pseudocode
        self.beta = "3*0.2/((self.N+1.3)**2+self.valP*self.popSize)"

        # expansion upon success (f_e line 7)
        self.ss = "1 + self.beta*(1-self.valP)"

        # Contraction f_c otherwise (line 8)
        self.sf = "1 - self.beta*(self.valP)"

        #learning rate rank-one update, when CMA
        #Note Matlab uses: CMA.ccov1 --> see if pendent nessecary here, why should adaption be off?
        #mueff hardcoded to one in Lp Adaption, mueff will be changed through the algorithm
        self.mueff = 1
        self.ccov1 = "3*0.2/((self.N+1.3)**2+self.mueff)"

        self.ccovmu = "min(1-self.ccov1, 3*0.2*(self.mueff-2+1/self.mueff) / ((self.N+2)**2+self.mueff*0.2))"

        #CMA learning constant for rank-one update
        self.cp = "1/np.sqrt(self.N)"

        #1: adapt hittin probability (to get a more accurate volume estimation or to get a better design center)
        # of interest if hitP_adapt == True
        self.hitP_adapt_cond = False

        if self.hitP_adapt_cond:
            self.hitP_adapt = {}
            self.hitP_adapt['Pvec']=[0.35,0.15,0.06,0.03,0.01]
            self.hitP_adapt['fixedSchedule'] = True
            self.hitP_adapt['maxEvalSchedule'] = [1/2,1/8,1/8,1/8,1/8]
            self.hitP_adapt['numLastSchedule'] = [1/2,3/4,3/4,3/4,3/4]
            self.hitP_adapt['testEvery'] = min(18/self.valP,self.maxMeanSize)

            self.hitP_adapt['stepSize'] = {'meanSize': "min(18 / self.valP, self.maxMeanSize)", 'deviation': 0.001}
            self.hitP_adapt['hitP'] = {'meanSize': "min(18 / self.valP, self.maxMeanSize)", 'deviation': 0.001}
            self.hitP_adapt['VolApprox'] = {'meanSize': "min(30 / self.valP, self.maxMeanSize)", 'deviation': 0.001}

            self.hitP_adapt['testStart'] = "max([2*self.hitP_adapt['stepSize']['meanSize']," \
                                           " 2*self.hitP_adapt['hitP']['meanSize']," \
                                           " 2*self.hitP_adapt['VolApprox']['meanSize']])"

            self.hitP_adapt['meanOfLast'] = 1/4
            self.hitP_adapt['deviation_stop'] = 0.01


    def adaptOption(self,inopts:dict):
        '''
        :param inopts: dictionary of options that differ to the default options, initialized above
        :return: dict with options
        '''
        opt_dict = self.__dict__
        for option,value in inopts.items():
            if option in opt_dict.keys():
                opt_dict[option] = value
            else:
                ValueError('Option %s is not an appropriate option for Lp-Adaption')
        return print(self.evaluateOpts(opt_dict))



    def evaluateOpts(self, optdict:dict):
        '''
        Function that evaluates all options given as a string and evaluates the equation behind them.
        :param optdict: dictionary of options
        :return: dictionaries, that contain evaluated parameters
        #TODO: Maybe options ccov1 and covvmu have to be adaptded concerning the equation. \
            Thus they should stay and evaluatable string
        '''

        for opt,value in optdict.items():
            if type(value) == str:
                optdict[opt] = eval(value)
            elif type(value) == dict:
                optdict[opt] = self.evaluateOpts(value)
            else:
                continue

        return optdict




