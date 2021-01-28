import numpy as np
import json
import re
import OptionHandler as oh
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

        # number of outputs from the oracle (first has to be 0/1), later more values possible
        self.nOut = 1

        # Plot results: for irst tests off, TODO: Set default 'on' if plotting is available
        self.plotting = False  #

        # After wich iteration one want to have a message from Commandline.
        # MaxEval % verbosemodulo == 0
        self.verboseModulo = int(1e3)

        # Save after i-th iteration, maxeval % savingmodulo ==0
        self.savingModulo = int(1e2)

        # Save data to file on/off
        self.bSaving = True

        # Save Covaraiance Matrices (leads to huge files)
        self.bSaveCov = True

        # save all numLast r, mu, Q, P_emp
        self.lastSaveAll = False

        # Save unfeasable Solutions
        self.unfeasibleSave = False

        #how many of numLast elements are used to get average mu and r
        self.numLast = oh.get_numLast(self.__dict__)

        #how many covariances should be used to get average covariance
        self.averageCovNum = 100

        # options concerning algorithmic parameters:
        #---------------------------------------------
        #Hitting Probability
        self.valP = 1 / np.exp(1)

        #  upper bound on inverval's size over which averaging happens ???
        self.maxMeanSize = 2000

        # size of moving window to get empirical hitting probability (in number of evaluations)
        self.windowSizeEval = oh.get_windowSizeEval(self.__dict__)

        # Initial Covariance
        self.r = 1

        #init Cholesky and Corvariance matrix
        self.initQ = np.eye(N).tolist()
        self.initC = np.eye(N).tolist()

        self.maxR = oh.get_maxR(self)

        self.minR = 0

        self.maxCond = 1e20*N

        #Mean apdaption weight
        self.N_mu = oh.get_N_mu(self.__dict__)

        #Matrix adaption weight
        self.N_C = ((N+1.3)**2+1)/2

    #TODO find out why gamma population size can be also np.floor(2 / self.valP) --> not in default mentioned in paper
        self.popSize = oh.get_Pop_size(self.__dict__)
        # Learning rate beta line 6 in pseudocode
        self.beta = oh.get_beta(self.__dict__)

        # expansion upon success (f_e line 7)
        self.ss = oh.get_ss(self.__dict__)

        # Contraction f_c otherwise (line 8)
        self.sf = oh.get_sf(self.__dict__)

        #learning rate rank-one update, when CMA
        #Note Matlab uses: CMA.ccov1 --> see if pendent nessecary here, why should adaption be off?
        #mueff hardcoded to one in Lp Adaption, mueff will be changed through the algorithm
        self.mueff = 1
        self.ccov1 = oh.get_ccov1(self.__dict__)
        self.ccovmu = oh.get_ccovmu(self.__dict__)

        #CMA learning constant for rank-one update
        self.cp = oh.get_cp(self.__dict__)

        #1: adapt hittin probability (to get a more accurate volume estimation or to get a better design center)
        # of interest if hitP_adapt == True
        self.hitP_adapt_cond = False

        #covariance adation
        self.cadapt = True

        # adapat mean
        self.madapt = True

        if self.hitP_adapt_cond:
            self.__invoke_hitP_adaption()



    def __invoke_hitP_adaption(self):
        self.hitP_adapt = {}
        self.hitP_adapt['pVec'] = [0.35, 0.15, 0.06, 0.03, 0.01]
        self.hitP_adapt['fixedSchedule'] = True
        self.hitP_adapt['maxEvalSchedule'] = [1 / 2, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
        self.hitP_adapt['numLastSchedule'] = [1 / 2, 3 / 4, 3 / 4, 3 / 4, 3 / 4]
        self.hitP_adapt['testEvery'] = oh.get_hitP_mean(self.__dict__)

        self.hitP_adapt['stepSize'] = {'meanSize':oh.get_stepSize_mean(self.__dict__),'deviation': 0.001}
        self.hitP_adapt['hitP'] = {'meanSize': oh.get_hitP_mean(self.__dict__), 'deviation': 0.001}
        self.hitP_adapt['VolApprox'] = {'meanSize': oh.get_volApprox_mean(self.__dict__), 'deviation': 0.001}

        self.hitP_adapt['testStart'] =oh.get_testStart(self.__dict__)

        self.hitP_adapt['meanOfLast'] = 1 / 4
        self.hitP_adapt['deviation_stop'] = 0.01

    def adaptOption(self,inopts:dict):
        '''
        :param inopts: dictionary of options that differ to the default options, initialized above
        :return: dict with options
        '''
        for option,value in inopts.items():
            if hasattr(self,option):
                setattr(self,option,value)
                # when adpation of hitting probability should be made, the class attributes has to be invoked
                # Note that the attributes will be invoked when given even if the cond is False
                if option == 'hitP_adapt_cond' and value == True or option == 'hitP_adapt' and value:
                    if option == 'hitP_adapt' and value and self.hitP_adapt_cond==False :
                        UserWarning('You want to initialize the options for the hitting pobability adaption, '
                                    'although condition is False. We set it for you. You might want to check, whether '
                                    'you want the options or not.')
                        self.hitP_adapt_cond = True

                    self.__invoke_hitP_adaption()
            if type(value) == str:
                setattr(self, option, re.sub('self\.','self\.opts\.',value))
            else:
                ValueError('Option %s is not an appropriate option for Lp-Adaption'%option)

        return self



    def evaluateOpts(self):
        '''
        Function that evaluates all options given as a string and evaluates the equation behind them.
        :param optdict: dictionary of options
        :return: dictionaries, that contain evaluated parameters
        #TODO: Maybe options ccov1 and covvmu have to be adaptded concerning the equation. \
            Thus they should stay and evaluatable string
        '''

        for opt,value in self.__dict__.items():
            if type(value) == str:
                setattr(self,opt,eval(value))
            elif type(value) == dict:
                attr = self.__resolve_dict(getattr(self,opt))
                setattr(self,opt,attr)
            else:
                continue

        return self

    def __resolve_dict(self,attr_dict):
        for key,value in attr_dict.items():
            if type(value) == str:
                attr_dict[key] = eval(value)
            elif type(value) == dict:
                attr_dict[key] = self.__resolve_dict(value)
            else:
                continue
        return attr_dict


