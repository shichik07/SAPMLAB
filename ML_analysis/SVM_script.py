# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:22:43 2020

@author: Julius Kricheldorff
"""

from scipy.io import loadmat
import os
import random
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

"""
Load Data
"""

os.getcwd()
os.chdir('C:\\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Analysis\Data')


for part in range(0,len(file)):
    file                      = os.listdir()
    p_dat                     = loadmat(file[part])
    p_dat.keys() # get keys for the imported dictionary data
    
    
    type(p_dat['labels']),p_dat['labels'].shape # data type and data size - one dim
    type(p_dat['trials']),p_dat['trials'].shape # data type and data size - three dim
    
    
    # labels bring labels in correct format - row one long row vector
    labels                    = np.squeeze(np.transpose(p_dat['labels'])) - 1
    # For the data, rows = examples, columns = features 
    trl_dat                   = p_dat['trials']
    
    '''
    Create Data array to safe variables
    '''
    
    my_df                     = pd.DataFrame(data=np.array(range(1,trl_dat.shape[2]+1)), 
                                             index=range(0,trl_dat.shape[2]), 
                                             columns=['Bin'])
    my_df['accuracy']         = np.zeros(trl_dat.shape[2])
    my_df['specificity']      = np.zeros(trl_dat.shape[2])
    my_df['sensitivity']      = np.zeros(trl_dat.shape[2])
    my_df['gamma']            = np.zeros(trl_dat.shape[2])
    my_df['C']                = np.zeros(trl_dat.shape[2])
    my_df['v_accuracy']       = np.zeros(trl_dat.shape[2])
    print(my_df)
                        
    
    """
   define the parameter space that will be searched over
    """
    
    C_range = np.logspace(-2, 10, 20, base =2)
    gamma_range = np.logspace(-9, 3, 20, base =2)
    param_grid = dict(gamma=gamma_range, C=C_range)
    
    '''
    initiate loop of time bins
    '''
    
    for t_bin in range(0,trl_dat.shape[2]): # time bins we are going to analyse
        data = trl_dat[:,:,t_bin]
        
        #create pipeline with two classifiers, logistic ridge regression and rbf-svm - 
        # important penalty parameter have to be fit seperately
        pipe = make_pipeline(
                svm.SVC(kernel='rbf',gamma = grid.best_params_['gamma'], 
                        C = grid.best_params_['C']),
                )
        
        #create stratified training and test splits 
        # NOTE: RANDOM STATE IS BE FIXED FOR EVERY PARTICIPANT 
        X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                            test_size=0.2, 
                                                            stratify=labels, 
                                                            random_state=part)
        
        
        
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