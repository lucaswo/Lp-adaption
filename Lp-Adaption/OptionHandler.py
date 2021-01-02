from typing import List,Dict
import numpy as np
'''
Collection of functions for parameters that have to be calculated from other options
'''

def get_numLast(parameters:Dict):
    num_last = max((parameters['maxEval'] - 1e3 * parameters['N']), parameters['maxEval'] * 0.7)
    return num_last

def get_windowSizeEval(parameters:Dict):
    windowSizeEval = min(110/parameters['valP'],parameters['maxMeanSize'])
    return windowSizeEval

def get_maxR(parameters:Dict):
    return np.inf

def get_ccov1(p:Dict):
    return 3*0.2/((p['N']+1.3)**2+p['mueff'])

def get_ccovmu(p:Dict):
    return min(1-p['ccov1'], 3*0.2*(p['mueff']-2+1/p['mueff']) / ((p['N']+2)**2+p['mueff']*0.2))

def get_N_mu(p:dict):
    return np.exp(1)*p['N']

def get_Pop_size(p:dict):
    return max(4 + np.floor(3 * np.log(p['N'])), np.floor(2 /p['valP']))

def get_beta(p:dict):
    3 * 0.2 / ((p['N'] + 1.3) ** 2 + p['valP'] * p['popSize'])