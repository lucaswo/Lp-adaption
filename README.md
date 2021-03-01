# Lp-Adaption
Algorithm from: J. Asmus, C. L. Mueller and I. F. Sbalzarini. Lp-Adaptation: Simultaneous 
Design Centering and Robustness Estimation of Electronic and Biological
Systems. Scientific Reports 2017

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

    def __init__(self, dim: int, pnorm: int, optsFile: str = '../Inputs/opts_example.json'):
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