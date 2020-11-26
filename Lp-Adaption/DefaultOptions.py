import numpy as np


class DefaultOptions:
    """
    This class handles all default parameters for the Lp_Adaption.
    All Parameters/options, which are not specified but necessary when starting the
    algorithm are taken from the default.
    All Options and their description can be additionally found in the README
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
        self.numLast = max((self.maxEval - 1e3*N),self.maxEval * 0.7)

        #how many covariances should be used to get average covariance
        self.averageCovNum = 100

        # options concerning algorithmic parameters:
        #---------------------------------------------
        #Hitting Probability
        self.valP = 1 / np.exp(1)

        #  upper bound on inverval's size over which averaging happens ???
        self.maxMeanSize = 2000

        # size of moving window to get empirical hitting probability (in number of evaluations)
        self.windowSizeEval = min(110/self.valP,self.maxMeanSize)

        # Initial Covariance
        self.r = 1

        #init Cholesky and Corvariance matrix
        self.initQ = np.eye(N)
        self.initC = np.eye(N)

        self.maxR = np.inf

        self.minR = 0

        self.maxCond = 1e20*N

        #Mean apdaption weight
        self.N_mu = np.exp(1)*N

        #Matrix adaption weight
        self.N_C = ((N+1.3)**2+1)/2

    #TODO find out why gamma population size can be also np.floor(2 / self.valP) --> not in default mentioned in paper
        self.popSize = max(4 + np.floor(3 * np.log(N)), np.floor(2 / self.valP))
        # Learning rate beta line 6 in pseudocode
        self.beta = 3*0.2/((N+1.3)**2+self.valP*self.popSize)

        # expansion upon success (f_e line 7)
        self.ss = 1 + self.beta*(1-self.valP)

        # Contraction f_c otherwise (line 8)
        self.sf = 1 - self.beta*(self.valP)

        #learning rate rank-one update, when CMA
        #Note Matlab uses: CMA.ccov1 --> see if pendent nessecary here, why should adaption be off?
        #mueff hardcoded to one in Lp Adaption, mueff will be changed through the algorithm
        self.mueff = 1
        self.ccov1 = 3*0.2/((N+1.3)**2+self.mueff)

        self.ccovmu = min(1-self.ccov1, 3*0.2*(self.mueff-2+1/self.mueff) / ((N+2)**2+self.mueff*0.2))

        #CMA learning constant for rank-one update
        self.cp = 1/np.sqrt(N)

        self.hitP_adapt = False

        if self.hitP_adapt:
            self.hitP_adapt_Pvec = [0.35,0.15,0.06,0.03,0.01]
            self.hitP_adapt_fixedSchedule = True
            self.hitP_adapt_maxEvalSchedule = [1/2,1/8,1/8,1/8,1/8]
            self.hitP_adapt_numLastSchedule = [1/2,3/4,3/4,3/4,3/4]
            #TODO rest parameters for adapting hitting probability!



