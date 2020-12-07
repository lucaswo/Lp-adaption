import numpy as np
from typing import List,Dict

class LpAdaption:

    def __init__(self,oracle:str,xstart:List,inopts:str):
        '''
        :param oracle: Python File containing oracle class with oracle function
        :param xstart: feasable point to start with
        :param inopts: input options
        '''

