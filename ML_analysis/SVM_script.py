# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:22:43 2020

@author: Julius Kricheldorff
"""

from scipy.io import loadmat
import os
import random
import matplotlib.pyprt as plt
import numpy as np

random.seed(1)
random.randint(1,10)


"""
Load Data
"""


os.getcwd()
os.chdir('C:\\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Analysis\Data')

file                = os.listdir()
p_dat               = loadmat(file[0])
p_dat.keys() # get keys for the imported dictionary data


type(p_dat['labels']),p_dat['labels'].shape # data type and data size - one dim
type(p_dat['trials']),p_dat['trials'].shape # data type and data size - three dim


# labels bring labels in correct format - row one long row vector
labels              = np.squeeze(np.transpose(p_dat['labels'])) - 1
# For the data, rows = examples, columns = features 
trl_dat             = p_dat['trials']

"""
Perform SVM classifaction
"""

from sklearn import svm
from sklearn.pipeline import make_pipeline

#create pipeline object
pipe = make_pipeline(
        svm.SVC(kernel='rbf', gamma = 0.000001, C = 0.001),
        )

# Note to self, for the classification folds need to be balanced - try random oversampling?

trl_dat_1           = trl_dat[:,:,59]
trl_dat_1_1         = trl_dat_1[:-10,:]
labels_1            = labels[:-10]
a = trl_dat_1_1[np.transpose(labels_1 == 1), :]
b = trl_dat_1_1[np.transpose(labels_1 == 0), :]
c = np.concatenate((a[0:40,:], b[0:40,:]), axis=0)
one = np.ones(40)
zero = np.zeros(40)
label_c = np.concatenate((one, zero), axis=0)

rbf_SVC             = svm.SVC(kernel='rbf', gamma = 0.000001, C = 0.001)

SVM = rbf_SVC.fit(c, label_c)
rbf_SVC.predict([trl_dat_1[-1,:]])
a = range(1,10)
total = 0
correct = 0
for i in range(1,20):
    total += 1
    
    print('The classifier predicted', rbf_SVC.predict([trl_dat_1[-i,:]]), 'the actual prediction was', labels[-i])
    if rbf_SVC.predict([trl_dat_1[-i,:]]) == labels[-i]:
        correct += 1
print('The classificationa ccuracy was: ', [correct/total])
# Check priors


print([trl_dat_1[-2,:]])