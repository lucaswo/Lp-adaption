import numpy as np
from typing import List, Dict
import re
from DefaultOptions import DefaultOptions
import json
from scipy.sparse.linalg import arpack
from Inputs import Oracles
from Vol_lp import vol_lp
from LpBallSampling import LpBall
from PlotData import PlotData
from numpy import matlib
import OptionHandler as oh


class LpAdaption:

    def __init__(self, xstart: List, inopts=''):
        '''

        :param oracle: Python File in Inputs directory
         containing oracle class with oracle function
        :param xstart: feasable point to start with
        :param inopts: input options
        '''

        self.oracle = Oracles.Oracle()
        # define dimension N trough starting point
        self.N = len(xstart)
        # load default options
        self.opts = DefaultOptions(self.N)
        self.xstart = xstart
        if inopts:
            try:
                if type(inopts) == str:
                    # swap default paramters in inopts
                    with open(inopts, 'r') as inopts_file:
                        inopts_dict = json.load(inopts_file)
                        self.opts.adaptOption(inopts=inopts_dict)
                else:
                    self.opts.adaptOption(inopts=inopts)
            except:
                ValueError('Look up your json inopts file for mistakes')
        # rename option strings, so they can be found from this class
        for opt, value in self.opts.__dict__.items():
            if type(value) == str:
                setattr(self.opts, opt, re.sub('self.', 'self.opts.', value))
        self.isbSavingOn = self.opts.bSaving
        self.isPlottingOn = self.opts.plotting

    def lpAdaption(self):
        # Test if hitP_adapt was set and the condition not, already caught when merging defopts and inopts
        # Dictionaray of recent parameters
        p = {}
        # __________SetUp of algorithmic parameters______________
        maxMeanSize = self.opts.maxMeanSize
        if self.opts.hitP_adapt_cond:
            p['valP'] = self.opts.hitP_adapt['pVec'][0]
        else:
            p['valP'] = self.opts.valP
        # initialize values for the parameters from the options
        p['nOut'] = self.opts.nOut
        p['maxEval'] = int(self.opts.maxEval)
        p['mueff'] = 1
        p['N_mu'] = self.opts.N_mu
        p['N_C'] = self.opts.N_C
        p['ccov1'] = self.opts.ccov1
        p['ccovmu'] = self.opts.ccovmu
        p['popSize'] = self.opts.popSize.astype('int')
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
        p['N'] = self.N

        if self.opts.hitP_adapt_cond:
            p['pVec'] = self.opts.hitP_adapt['pVec']
            p['lpVec'] = len(p['pVec'])
            # TODO Implement adaptable hitting probability rest
        else:
            numLast = self.opts.numLast
            # Check, that numLast is smaller than MaxEval
            if numLast > p['maxEval']:
                numLast = max((p['maxEval'] - 1e3 * self.N), p['maxEval'] * 0.7)
                Warning('Num Last is too big, it was changed to default %d.' % numLast)

        averageCoNum = self.opts.averageCovNum

        Q = np.array(self.opts.initQ)
        C = np.array(self.opts.initC)
        # C consistent calculated from Q for compairison with specified C
        C_calc = p['r'] ** 2 * (Q * Q)
        np.testing.assert_equal(C, C_calc, err_msg='Initialized C Matrix is not consistent with Matrix Q.\n'
                                                   'Your Q yields to the following C:' % C_calc)
        # check if C is positiv semidefinit
        # TODO: Maybe easier way
        [Bo, tmp] = np.linalg.eig(C)
        tol = 1e-8
        vals, vecs = arpack.eigsh(C, k=2, which='BE')
        if not np.all(vals > -tol):
            ValueError('Covariance Matrix need tobe positiv semidefinit!')
        elif np.size(C, 1) != self.N:
            ValueError(' C has not the same size as your starting point xstart!')

        diagD = np.sqrt(np.diag(tmp))
        detdiagD = np.prod(diagD)
        diagD = diagD / (detdiagD ** (1 / self.N))
        Q = Bo * (np.tile(diagD, (self.N, 1)))
        r = np.linalg.det(C) ** (1 / 2 * self.N)
        # matlab checks for the same size of C and xstart here again
        # Anyway, we not cause the check is already above and not in an if/else statement(cause initC and
        # Q are always provided through default Options)

        [j, eigVals] = np.linalg.eig(Q * np.transpose(Q))
        condC = np.linalg.cond(Q * np.transpose(Q))

        # ___________Setup initial settings_____________
        # check if xstart is a feasable point
        xstart_out = self.oracle.oracle(self.xstart)
        len_x = len(xstart_out)
        # if dim of vector xstart and nOut Parameter differs, one might want to check the oracle
        if len_x != p['nOut']:
            UserWarning('Dimension of Oracle output differs from nOut option! nOut will be set so the len of xStart!')
            p['nOut'] = len_x
        if xstart_out[0] != 1:
            ValueError('x_start needs to be a feasable point')

        lastxAcc = self.xstart
        # Number of function evaluations
        counteval = [1]
        if self.opts.hitP_adapt_cond:
            vcounteval = [1]
            vcountgeneration = 1

            if self.opts.hitP_adapt['fixedSchedule']:
                cntsave_Part = 1
        else:
            # Number of evaluations after MaxEval - numLast evaluations
            countevalLast = 0
            # Number of accepted points after MaxEval - numLast points
            lastNumAcc = 0

        countgeneration = 1

        # Number of all accepted points equals one because we need to start with a feasible point!
        numAcc = 1
        # Number of accepted points for specific hitP
        vNumAcc = 1
        mu = np.array(self.xstart)

        # ___________Setup Output Parameters_____________
        if self.isbSavingOn:
            xRawDim = (np.ceil(p['maxEval'] / self.opts.savingModulo).astype('int'), self.N)
            xRaw = np.empty(shape=xRawDim)
            xRaw[0, :] = self.xstart
            # save all accepted x to estimate the upper bound of the volume
            xAcc = np.empty((int(p['maxEval']), self.N))
            # counteval of all accepted x
            cntAcc = np.empty(shape=(int(p['maxEval']), 1))

            cntAcc[0] = 1

            # oracle output is a vector out of 0s and 1s
            if p['nOut'] > 1:
                fxAcc = np.empty(shape=(int(p['maxEval']), p['nOut'] - 1))
                fxAcc[0, :] = xstart_out[1:]
            xAcc[0, :] = np.transpose(xstart_out)

            if self.opts.unfeasibleSave:
                xNotAcc = np.empty(shape=(int(p['maxEval']), p['nOut'] - 1))
                cntNotAcc = np.empty(shape=(p['maxEval'], 1))
                if p['nOut'] > 1:
                    fxNotAcc = np.empty(shape=(int(p['maxEval']), p['nOut'] - 1))

            # Vector, if sample was accepted or not
            c_TVec = np.empty(shape=(np.ceil(p['maxEval'] / self.opts.savingModulo).astype('int'), 1))
            c_TVec[0] = xstart_out[0]

            # if output length of oracle is bigger than one, the following oracle values have to be saved
            if p['nOut'] > 1:
                fc_TVec = np.empty(shape=(np.ceil(p['maxEval'] / self.opts.savingModulo).astype('int'), p['nOut'] - 1))
                fc_TVec[0, :] = xstart_out[1:]

            # Vector of evaluation indices when everything is saved, first one is one, because xstart is feasable point
            # TODO: Ist Vector für die Gesamtentscheidungs-Speicherung?
            countVec = np.empty(shape=(np.ceil(p['maxEval'] / self.opts.savingModulo).astype('int'), 1))
            countVec[0] = 1

            # TODO: in Matlab just a 1x1 cell for double value, init. here usefull?
            stopFlag = None

            # settings for CMA/GaA
            verboseModuloGen = np.ceil(self.opts.verboseModulo / p['popSize']).astype('int')
            savingModuloGen = np.ceil(self.opts.savingModulo / p['popSize']).astype('int')
            tmp_num = np.ceil(p['maxEval'] / savingModuloGen).astype('int')
            if self.opts.hitP_adapt_cond:
                # how often hitting probability is changed
                cntAdapt = 1
                # save different values for parameter when hitp changes
                n_MuVec = np.empty(shape=(p['lpVec'], 1))
                n_MuVec[0] = p['N_mu']
                betaVec = np.empty(shape=(p['lpVec'], 1))
                betaVec[0] = p['beta']
                ssVec = np.empty(shape=(p['lpVec'], 1))
                ssVec[0] = p['ss']
                sfVec = np.empty(shape=(p['lpVec'], 1))
                sfVec[0] = p['sf']
                iterVec = np.empty(shape=(p['lpVec'], 1))
                windowSize = np.ceil(self.opts.windowSizeEval / p['popSize'])
                ccov1Vec = np.empty(shape=(p['lpVec'], 1))
                ccov1Vec[0] = p['ccov1']
                ccovmu_Vec = np.empty(shape=(p['lpVec'], 1))
                ccovmu_Vec[0] = p['ccovmu']
                cntGenVec = np.empty(shape=(p['lpVec'], 1))
                popSizeVec = np.empty(shape=(p['lpVec'], 1))
                popSizeVec[0] = p['popSize']

            alpha_p = 1
            # initial evolution path for C
            pc = np.zeros(shape=(self.N, 1))
            # Save generation of of accepted samples
            cntAccGen = np.empty(shape=(p['maxEval'], 1))
            # Vector of evaluation indices when r,mu is saved
            cntVec = np.empty(shape=(tmp_num, 1))
            cntVec[0] = 1

            # TODO Find out if next line of code (comment) is needed, matlab code not sure
            saveIndGeneration = 2

            # Vector of step length
            rVec = np.zeros(shape=(tmp_num, 1))
            rVec[0] = p['r']

            # Vector of mu
            muVec = np.empty(shape=(tmp_num, self.N))
            muVec[0, :] = mu

            # Vector of Volumina of the Lp Balls
            volVec = np.zeros(shape=(tmp_num, 1))
            volVec[0] = np.abs(np.linalg.det(Q)) * vol_lp(self.N, p['r'], p['pn'])

            # Vector of empirical acceptance probability
            p_empVecAll = np.zeros(shape=(tmp_num, 1))
            p_empVecAll[0] = p['valP']
            p_empVecWindow = np.zeros(shape=(tmp_num, 1))
            p_empVecWindow[0] = p['valP']
            numMuVec = np.zeros(shape=(p['windowSize'], 1))

            # Cell array of Q matrices
            # TODO: test in the end if len of qcell list is tmp_num
            if self.opts.bSaveCov:
                qCell = []
                qCell.append(Q)

            if self.opts.hitP_adapt_cond:
                # for each hitP save mu. r, hitP, Q
                muLastVec = np.empty(shape=(p['lpVec'], 1))
                rLastVec = np.empty(shape=(p['lpVec'], 1))
                qLastCell = []  # shape = cell(lPVec,1)

                hitPLastVec = np.empty(shape=(p['lpVec'], 1))
                sizeLastVec = np.empty(shape=(p['lpVec'], 1))
                approxVolLastVec = np.empty(shape=(p['lpVec'], 1))

                if self.opts.hitP_adapt['fixedSchedule']:
                    maxEval_Part = np.floor(self.opts.hitP_adapt['maxEvalSchedule'][cntAdapt]) * p['maxEval']
                    numLastPart = np.ceil(
                        self.opts.hitP_adapt['numLastSchedule'][cntAdapt] * p['maxEval'] / p['popSize'])

                    muLast = np.empty(shape=(numLastPart.astype('int'), self.N))
                    rLast = np.zeros(shape=(np.ceil(numLastPart).astype('int'), 1))
                    p_empLast = np.empty(shape=(np.ceil(numLastPart).astype('int'), 1))

            else:
                if numLast:
                    # Trace of numLast mu, r and hittingP values
                    muLast = np.empty(shape=(np.ceil(numLast / p['popSize']).astype('int'), self.N))
                    rLast = np.zeros(shape=(np.ceil(numLast / p['popSize']).astype('int'), 1))
                    p_empLast = shape = (np.ceil(numLast / p['popSize']).astype('int'), 1)

        if self.opts.hitP_adapt_cond:
            # save alle step sizes
            rVec_all = np.empty(shape=(np.ceil(p['maxEval'] / p['popSize']).astype('int'), 1))
            rVec_all[0] = p['r']
            # save alle hitP
            hitP_all = np.empty(shape=(np.ceil(p['maxEval'] / p['popSize']).astype('int'), 1))
            hitP_all[0] = p['valP']
            # save corresponding function evaluations
            cnt_all = np.empty(shape=(np.ceil(p['maxEval'] / p['popSize']).astype('int'), 1))
            cnt_all[0] = 1
        # gets 1 if change of hitP occurs TODO: relevant, notwendig?
        van = 0
        saveInd = 2
        saveIndAcc = 2
        saveIndNotAcc = 1  # everything starts with feasable point
        stopFlag = ''
        if not (self.opts.hitP_adapt_cond and not self.opts.hitP_adapt['fixedSchedule']):
            saveIndLast = 0  # number of iterations after max eval - numLast evaluations
        if self.opts.hitP_adapt_cond and self.opts.hitP_adapt['fixedSchedule']:
            MaxEval_Part = np.floor(self.opts.hitP_adapt['maxEvalSchedule'][cntAdapt] * p['maxEval'])
            numLast_Part = np.floor(self.opts.hitP_adapt['numLastSchedule'][cntAdapt] * MaxEval_Part)

        # _______________________Generation Loop_______________________

        PopSizeOld = []
        lastEval = 1

        [Bo, tmp] = np.linalg.eig(C)
        diagD = np.sqrt(np.diag(tmp))

        invB = np.diag(1 / diagD) * Bo
        while counteval[-1] < (p['maxEval'] - p['popSize'] - lastEval):
            counteval = np.array(range(1, p['popSize'].astype('int'))) + counteval[-1]

            if self.opts.hitP_adapt_cond:
                vcounteval = np.array(range(1, p['popSize'].astype('int'))) + vcounteval[-1]
                vcountgeneration += 1

            countgeneration += 1

            # generate PopSize candidate solutions from LpBall sampling
            if p['pn'] > 100:  # sample uniform from hypercube with radius 1
                arz = 2 * np.random.rand((self.N, p['popSize'])) - 1
            else:
                arz = np.transpose(LpBall(dim=self.N, pnorm=p['pn']).samplefromball(number=p['popSize']))

            # sampled vectors as input for oracle
            v = np.transpose(np.tile(mu, (p['popSize'].astype('int'), 1)))
            arx = np.add(v,r*(Q @ arz))

            if self.isPlottingOn and self.isbSavingOn:
                plot = PlotData()
                plot.plot(cntVec, countgeneration, saveIndGeneration, muVec, rVec, verboseModuloGen, self.N, eigVals, r,
                          p_empVecAll, p_empVecWindow, p['p_empAll'], p['p_empWindow'])

            # ________Oracle_______________
            # vector of oracle decisions for Popsize samples
            c_T = np.zeros(shape=(p['popSize'].astype('int'), 1))
            # Matrix of oracle outputs for every sample
            outArgsMat = np.empty(shape=(p['popSize'].astype('int'), p['nOut'] - 1))

            for s in range(0,p['popSize']):
                outArgs = self.oracle.oracle(arx[:,s])
                c_T[s] = outArgs[0]
                if p['nOut']>1:
                    if np.any(outArgs):
                        outArgsMat[s,:] = None
                    else:
                        outArgsMat[s,:] = outArgs[1:]

            # numfeas candidate solutions are in feasable region
            numfeas = np.sum(c_T==1)
            numMuVec[(countgeneration % p['windowSize'])] = numfeas
            numAccWindow = sum(numMuVec)

            if numfeas >0:
                #get alle feasable point from candidate solutions
                indexes = np.where(c_T==1)[0]
                pop = arx[:,indexes]
                weights = np.ones(shape=(numfeas,1))/numfeas # uniform weights

                #count accepted solutions
                numAcc = numAcc +numfeas
                if self.opts.hitP_adapt_cond:
                    vNumAcc = vNumAcc + numfeas

            if self.opts.hitP_adapt_cond:
                rVec_all[countgeneration] =r
                #TODO: Immplement hittingP Adaption

            if van == 0:
                p['mueff'] = numfeas
                p['ccov1'] = oh.get_ccov1(p)
                p['ccovmu'] = oh.get_ccovmu(p)
                p['N_mu'] = oh.get_N_mu(p)

            #____________ %% Code for adaptation of C based on original Adaptive Encoding
        #   procedure by N. Hansen, see
        #    REFERENCE: Hansen, N. (2008). Adaptive Encoding: How to Render
        #    Search Coordinate System Invariant. In Rudolph et al. (eds.)
        #    Parallel Problem Solving from Nature, PPSN X,
        #    Proceedings, Springer. http://hal.inria.fr/inria-00287351/en/
        #_____________________________________________________________________________
            #Adapt step size, (ball radius r)
            # Depends on the number of feasable Points,
            # Pseudo code line 15
            r = r*p['ss']**numfeas * p['sf']**(p['popSize']-numfeas)
            r = max(min(r,p['rMax']),p['rMin'])

            # Adapt mu
            mu_old = mu
            # if feasable points found, adapt mean and finally proposal distribution...
            if numfeas != 0 and self.opts.madapt:
                #adapt mean, pseudo code line 17
                mu = (1-1/p['N_mu'])*mu +np.reshape((1/p['N_mu'])*pop @ weights,(2,))

                #Update evolution paths
                if sum((invB*(mu-mu_old))**2).any() ==0:
                    z=0
                else:
                    alpha0 = p['l_expected']/np.sqrt(sum((invB*(mu-mu_old))**2))
                    #part of pseudo code line 19 alpha0(mu-mu_old), calculation of alph_j see Nikolaus,Hansen paper 2008
                    z = alpha0*(mu-mu_old)
            else:
                z = 0

            pc = (1-p['cp'])*pc +np.sqrt(p['cp']*(2-p['cp']))*z
            s = pc * np.transpose(pc)





lp = LpAdaption(xstart=[1, 1])
lp.lpAdaption()
