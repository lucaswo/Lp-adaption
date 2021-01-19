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