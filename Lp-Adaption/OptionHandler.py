from typing import List,Dict
import numpy as np
'''
Collection of functions for parameters that have to be calculated from other options
Any parameter, that has to be evaluated from others, should be mentioned here
'''
#TODO: if the user wants to use a new function, then he/she has to declare the function here and in the opts json with str 'optionhandler'
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
    np.seterr(divide='ignore', invalid='ignore')
    return min(1-p['ccov1'], np.divide(3*0.2*np.divide(p['mueff']-2+1,p['mueff']) , ((p['N']+2)**2+p['mueff']*0.2)))

def get_N_mu(p:dict):
    return np.exp(1)*p['N']

def get_Pop_size(p:dict):
    return max(4 + np.floor(3 * np.log(p['N'])), np.floor(2 /p['valP']))

def get_beta(p:dict):
    return 3 * 0.2 / ((p['N'] + 1.3) ** 2 + p['valP'] * p['popSize'])

def get_ss(p:dict):
    return 1 + p['beta']*(1-p['valP'])

def get_sf(p:dict):
    return 1 - p['beta']*(p['valP'])

def get_cp(p:dict):
    return 1/np.sqrt(p['N'])

def get_oracle_inopts(p:dict):
    return []

#____________Set up Functions for Hitting probability adaption_____________

def get_stepSize_mean(p:dict):
    return min(18 / p['valP'], p['maxMeanSize'])

def get_hitP_mean(p:dict):
    return min(30 / p['valP'], p['maxMeanSize'])

def get_hitP_testEvery(p:dict):
    return min(18 / p['valP'], p['maxMeanSize'])

def get_volApprox_mean(p:dict):
    return min(18 / p['valP'], p['maxMeanSize'])

def get_testStart(p:dict):
    return max([2*p['hitP_adapt']['stepSize']['meanSize'],
                2*p['hitP_adapt']['hitP']['meanSize'],
                2*p['hitP_adapt']['VolApprox']['meanSize']])