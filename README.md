# Lp-Adaption
Algorithm from: J. Asmus, C. L. Mueller and I. F. Sbalzarini. Lp-Adaptation: Simultaneous 
Design Centering and Robustness Estimation of Electronic and Biological
Systems. Scientific Reports 2017

## Requirements
The Lp-Adaption needs just a few modules, that can be found here in the `requirements.txt`. You can istall them via 
pip in your selected virtualenv.
```shell script
pip install -r requirements.txt 
```

## Usage of Lp-Adaption

The Lp-adaption needs an oracle a pnorm and a feasible starting point and the parameter options to calculate a 
design center.
The oracle has to be a python function, the starting point a vector in the dimension N and the parameters are given
 as a json file or dictionary for constants and a python file of functions for calculated and recalculated values.
 
 ```python
inopts = 'inopts.json'
xstart = [1,1]
LpAdaption(oracle,xstart,inopts)
```
The following parameters have functions instead of constants. 
Therefore they have to be adapted in the `OptionHandler.py`

## Exectuting Examples
When running one of the examples in the 'Examples' folder, make sure that your working directory is the 'Lp-Adaption' 
directory. Otherwise, reference files can't be found 

## Writing your own Lp Adaption 
It's recommended, to init a new class with your oracle as a class function. 
A template:

 ```python
import numpy as np
import json
import Vol_lp
import LpBallSampling
import LpAdaption


class LpBallExample():

    def __init__(self, dim: int, pnorm: int, optsFile: str = '../Inputs/example_lpball.json'):
        with open(optsFile) as file:
            self.optsDict = json.load(fp=file)
        self.dim = dim
        self.pn = pnorm

    def lp_adaption(self):
        xstart = [1,-1,1]
        l = LpAdaption.LpAdaption(self.oracle, xstart, inopts=self.optsDict)
        out = l.lpAdaption()

    def oracle(self, x,inopts):
       return sum(x) >= 1

l = LpBallExample(dim=3,pnorm=2)
l.lp_adaption()
```
Note. that there are some Parameters which are calculated and recalculated through the Lp adaption.
You can change these parameters in the `OptionHandler.py` file.

## Changable Parameters
There are two types of parameters one can adapt for the algorithm. Firstly the  constant values. They can be given to the algorithm as a Dictionary
 or a `.json` file respectively.Secondly the 
calculated values, which may change through the algorithm. Therefore the second type needs a python file with functions 
for the calculation (`OptionHandler.py`).

### List of constant values
An Example for such a json file can be found in the Inputs directory. You can use it directly as a template.
The default Parameters can be found in the `options_default.json` in the input Directory
The following Parameters are defined:
- **"N"**: Int, Dimension
- **"maxEval"**: Int maximum number of function evaluations
- **"pn"**: Int, pnorm 
- **"nOut"**: Int, output dim
- **"plotting"**: Bool, Plotting on or off
- **"verboseModulo"**: After which iteration one want to have a message from Commandline. (MaxEval % verbosemodulo == 0)
- **"savingModulo"**: int, Save after i-th iteration, maxeval % savingmodulo ==0
- **"bSaving"**: Bool, Save data to file on/off
- **"bSaveCov"**: Bool, Save Covaraiance Matrices
- **"lastSaveAll"**: save all numLast r, mu, Q, P_emp
- **"unfeasibleSave"**: Bool, save unfeasable Points as well
- **"averageCovNum"**: Int, how many of numLast elements are used to get average mu and r
- **"valP"**:float, Hitting Probability
- **"maxMeanSize"**: Int, upper bound on inverval's size over which averaging happens
- **"r"**: float, starting radius
- **"initQ", List, Starting Q Matrix
- **"initC"**:list, Starting C Matrix
- **"maxR"**: "np.inf", Maximum radius
- **"minR"**: 0, min Radius
- **"maxCond"**: 2e+20, maximal allowed condition
- **"N_mu"**: Mean adaptation weight
- **"N_C"**: Matrix adaptation weight
- **"hitP_adapt_cond"**: Bool, If hitting probability is adapted or not 
- **"hitP_adapt"**: 
    - **"Pvec"**: list, hitting Probability adaption values
    - **"fixedSchedule"**: Bool,
    - **"maxEvalSchedule"**: proportion of maxEval
    - **"numLastSchedule"**: over how many samples of each run should be averaged to get radius r and mean mu of found feasible region 
    (interest if hitP_adapt ==1 and no fixed schedule)
    - **"testEvery"**: every which iteration it should be tested if step size is in steady state
    - **"stepSize"**: 
        - **"deviation"**: float at which deviation to start stepsize adaption
    - **"hitP"**: 
        - **"deviation"**: float
    - **"VolApprox"**:
        - **"deviation"**: float
    - **"meanOfLast"**: float, defining how many samples of each run are used for calculating the average if no fixed schedule for the changing of the hitting probability
    (number between 0 and 1)
    - **"deviation_stop"**: double

## Calculated Values
List of Calculated Values with their defaults:
- **numLast**:
    ```python 
    num_last = max((p['maxEval'] - **1e3 * p['N']), p['maxEval'] * 0.7)
    ```

- **windowSizeEval**:
    ```python 
    windowSizeEval = min(110/p['valP'],p['maxMeanSize'])
    ```

- **maxR**:
    ```python 
    np.inf
    ```

- **ccov1**:
    ```python 
    return 3*0.2/((p['N']+1.3)**2+p['mueff'])
    ```
- **ccovmu**:
    ```python 
    np.seterr(divide='ignore', invalid='ignore')
    return min(1-p['ccov1'], np.divide(3*0.2*np.divide(p['mueff']-2+1,p['mueff']) , ((p['N']+2)**2+p['mueff']*0.2)))
    ```
- **N_mu**:
    ```python 
    return np.exp(1)*p['N']
    ```
- **Pop_size**:
    ```python 
    return max(4 + np.floor(3 * np.log(p['N'])), np.floor(2 /p['valP']))
    ```
- **beta**:
    ```python 
    return 3 * 0.2 / ((p['N'] + 1.3) ** 2 + p['valP'] * p['popSize'])
    ```
- **ss**:
    ```python 
    return 1 + p['beta']*(1-p['valP'])
    ```
- **sf**:
    ```python 
    return 1 - **p['beta']*(p['valP'])
    ```
- **cp**:
    ```python 
    return 1/np.sqrt(p['N'])
    ```
- **oracle_inopts**:
    ```python 
    return []
    ```

- **stepSize_mean**:
    ```python 
    return min(18 / p['valP'], p['maxMeanSize'])
    ```
- **hitP_mean**:
    ```python 
    return min(30 / p['valP'], p['maxMeanSize'])
    ```
- **hitP_testEvery**:
    ```python 
    return min(18 / p['valP'], p['maxMeanSize'])
    ```
- **volApprox_mean**:
    ```python 
    return min(18 / p['valP'], p['maxMeanSize'])
    ```
- **testStart**:
    ```python 
    return max([2*p['hitP_adapt']['stepSize']['meanSize'],
                2*p['hitP_adapt']['hitP']['meanSize'],
                2*p['hitP_adapt']['VolApprox']['meanSize']])
    ```

