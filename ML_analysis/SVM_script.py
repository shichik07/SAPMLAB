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
import logging


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
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


#create pipeline object
pipe = make_pipeline(
        svm.SVC(kernel='rbf', gamma = 0.000001, C = 0.001)
        )

# Note to self, for the classification folds need to be balanced - try random oversampling?

trl_dat_1           = trl_dat[:,:,7]
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
'''
brief outline of my pipeline
'''

# define the parameter space that will be searched over
C_range = np.logspace(-2, 10, 20, base =2)
gamma_range = np.logspace(-9, 3, 20, base =2)
param_grid = dict(gamma=gamma_range, C=C_range)


for part in participants:
    
    for t_bin in bins:



#create pipeline with two classifiers, logistic ridge regression and rbf-svm - 
# important penalty parameter have to be fit seperately
pipe = make_pipeline(
        svm.SVC(kernel='rbf',gamma = grid.best_params_['gamma'], 
                C = grid.best_params_['C']),
        )

#create stratified training and test splits NOTE: RANDOM STATE MUST BE FIXED FOR EVERY PARTICIPANT ANEW
X_train, X_test, y_train, y_test = train_test_split(trl_dat_1, labels, 
                                                    test_size=0.2, 
                                                    stratify=labels, 
                                                    random_state=0)



 # now create a searchCV object and fit it to the data
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), 
                    param_grid=param_grid, cv=cv, refit = True)
grid.fit(X_train, y_train)


grid.predict(X_test)
"""
Note for the grid-predict scoring. Goes to show that just making sure that all
of our classes are equally represented in the training and valdiation samples is
not a good strategy. Results indicate that we build a classifier that simply 
rates all available examples as the majority class and by that achieves a higher
classification accuracy. Strategy I intend to follow instead is undersampling 
with a customade script. NOTE: setting classweight to 'balanced' did do no good 
either, same classifier behavior.
"""

#fit the models
pipe.fit(X_train, y_train)
Pipeline(steps=[('support', svm.SVC(kernel='rbf')])

#evaluate outcome
accuracy_score(pipe.predict(X_test), y_test)



"""
How to fit C-regeuralization parameter and gamma-training example influence 
parameter
"""
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-2, 10, 20, base =2)
gamma_range = np.logspace(-9, 3, 20, base =2)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(trl_dat_1, labels)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Now we need to fit a classifier for all parameters in the 2d version
# (we use a smaller set of parameters here because it takes a while to train)
grid.best_params_['C']


pipe = make_pipeline(
        svm.SVC(kernel='rbf',gamma = grid.best_params_['gamma'], C = grid.best_params_['C']),
        )