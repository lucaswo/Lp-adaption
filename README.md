# L<sub>p</sub>-Adaption
Python implementation of the MATLAB Version from:  
[Asmus, J., Müller, C.L. & Sbalzarini, I.F. L<sub>p</sub>-Adaptation: Simultaneous Design Centering and Robustness Estimation of Electronic and Biological Systems. _Sci Rep_ 7, 6660 (2017).](https://www.nature.com/articles/s41598-017-03556-5)

## Requirements
The L<sub>p</sub>-Adaption needs just a few modules that can be found here in the `requirements.txt`. You can install them via 
pip in your selected virtualenv.
```shell script
pip3 install -r requirements.txt 
```

## Usage of L<sub>p</sub>-Adaption

The L<sub>p</sub>-Adaption needs an oracle, a pnorm, a feasible starting point, and the parameter options to calculate a 
design center. The oracle has to be a python function, the starting point a vector with N dimensions, and the parameters are given as a json file for constants and a python file of functions for calculated and recalculated values.
 
 ```python
inopts = 'inopts.json'
xstart = [1,1]
LpAdaption(oracle,xstart,inopts)
```

## Executing Examples
When running one of the examples in the 'Lp-Adaption/Examples' folder, make sure that your working directory is the 'Lp-Adaption' 
directory. Otherwise, referenced files can't be found. 

## Writing your own L<sub>p</sub>-Adaption 
It's recommended, to initialize a new class with your oracle as a class function. 
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
Note. that there are some Parameters which are calculated and recalculated through the L<sub>p</sub>-Adaption.
You can change these parameters in the `OptionHandler.py` file.

## Changeable Parameters
There are two types of parameters one can adapt for the algorithm. First, there are the constant values. They can be given to the algorithm as a dictionary or a `.json` file respectively. the second type of parameters are the calculated values, which may be used during run time by the algorithm. Therefore, the second type needs a python file with functions for the calculation (`OptionHandler.py`).

### Constant Values
An example for a json file with constant values can be found in the 'Inputs' directory. You can use it directly as a template.
The default parameters can be found in `options_default.json` or  `DefaultOptions.py`.
The following Parameters are defined:
- **"N"**: Int, Dimension
- **"maxEval"**: Int maximum number of function evaluations
- **"pn"**: Int, pnorm 
- **"nOut"**: Int, output dim
- **"plotting"**: Bool, plotting on or off
- **"verboseModulo"**: log every i-th iteration, MaxEval % verboseModulo == 0
- **"savingModulo"**: int, save every i-th iteration, MaxEval % savingModulo ==0
- **"bSaving"**: Bool, save data to file
- **"bSaveCov"**: Bool, save covariance matrices
- **"lastSaveAll"**: save all numLast, r, mu, Q, P_emp
- **"unfeasibleSave"**: Bool, save unfeasible points
- **"averageCovNum"**: Int, how many of numLast elements are used to get average mu and r
- **"valP"**: float, hitting probability
- **"maxMeanSize"**: Int, upper bound for interval over which averaging happens
- **"r"**: float, starting radius
- **"initQ"**, list, starting Q Matrix
- **"initC"**: list, starting C Matrix
- **"maxR"**: "np.inf", maximum radius
- **"minR"**: 0, min radius
- **"maxCond"**: 2e+20, maximal allowed condition
- **"N_mu"**: mean adaptation weight
- **"N_C"**: matrix adaptation weight
- **"hitP_adapt_cond"**: Bool, if hitting probability is adapted or not 
- **"hitP_adapt"**: 
    - **"Pvec"**: list, hitting probability adaption values
    - **"fixedSchedule"**: Bool, use fixed schedule for hitting probability adaption
    - **"maxEvalSchedule"**: proportion of maxEval for schedule
    - **"numLastSchedule"**: over how many samples of each run should be averaged to get radius r and mean mu of found feasible region 
    (if hitP_adapt == 1 and no fixed schedule)
    - **"testEvery"**: every i-th iteration it should be tested if step size is in steady state
    - **"stepSize"**: 
        - **"deviation"**: float at which deviation to start step size adaption
    - **"hitP"**: 
        - **"deviation"**: float at which deviation to start hitting probability adaption
    - **"VolApprox"**:
        - **"deviation"**: float at which deviation to start volume approximation adaption
    - **"meanOfLast"**: float, defining how many samples of each run are used for calculating the average if there is no fixed schedule for the changing of the hitting probability
    (between 0 and 1)
    - **"deviation_stop"**: float stop criteria

### Calculated Values
List of calculated values with their defaults:
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

