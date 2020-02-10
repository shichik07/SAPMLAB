# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:22:43 2020

@author: Julius Kricheldorff
"""

from scipy.io import loadmat
import os
import random

random.seed(1)
random.randint(1,10)


"""
Load Data
"""


os.getcwd()
os.chdir('C:\\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\
         Signal Processing Course\Experiment and Analysis\Analysis\Data')
file                = os.listdir()
p_dat               = loadmat(file[0])
p_dat.keys() # get keys for the imported dictionary data


type(p_dat['labels']),p_dat['labels'].shape # data type and data size - one dim
type(p_dat['trials']),p_dat['trials'].shape # data type and data size - three dim

labels              = p_dat['labels']
trl_dat             = p_dat['trials']

"""
Perform SVM classifaction
"""