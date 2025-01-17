# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:22:25 2020

@author: juliu
"""

%reset
from scipy.io import loadmat
import os
import re
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
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import RidgeClassifierCV

"""
Sensitivity and Specificity calculation
"""

def perf_measure(y_actual, y_hat):
    # Partial Source: 
    #//stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    accu = round((TP+TN)/len(y_hat),4) #accuracy = TP+TN /all
    sens = round(TP/(TP+FN),4) # Sensitivity = true positives over all positives
    spec = round(TN/(TN+FP),4) # specificity = true negatives over all negatives
    prec = round(TP/(TP+FP),4) # precision = true positives over all predicted positives
    F1   = round((2*TP)/(2*TP+FP+FN),4) # harmonic mean of precision and sensitivity
    return(accu, sens, spec, F1, prec)

def oversampling(X, y, part):
    # function determines if classes are imbalanced - if so random oversampling
    # with replacement is performed and a new label and data array is outputed. 
    # We do not save the label array, but the seed of the random generator, so 
    # that we get the same random sequence everytime this script is run 
    # and the analysis is computationally replicable. NOTE we do not shuffle the 
    # returned data and labels because that is already done for us when splitting
    # into training and validation
    
    # initialize random seed for re-sampling
    np.random.seed(part*17)
   
     
    # get indexes of label classes
    zeros = y==0
    ones  = y==1
    
    if sum(ones) is sum(zeros):
        return X,y
    else:
        if sum(zeros) > sum(ones):
            rds = np.random.choice(sum(ones), sum(zeros)-sum(ones), replace=True)
            sel = np.flatnonzero(ones) # get indexes ones
            sel = sel[rds] # get redrawn indexes
            add = X[sel,:,:]
            X   = np.append(X, add, 0) # update trials
            y   = np.append(y, np.ones(len(rds))) #updated labels
            
            if sum(y==1) != sum(y==0): #double check
                raise NameError('Oversampling failed!')
            else:
                print('Data has successfully been oversampled!')
                return X,y
        else:
            rds = np.random.choice(sum(zeros), sum(ones)-sum(zeros), replace=True)
            sel = np.flatnonzero(zeros) # get indexes zeros
            sel = sel[rds] # get redrawn indexes
            add = X[sel,:,:]
            X   = np.append(X, add, 0) # update trials
            y   = np.append(y, np.zeros(len(rds))) #updated labels
            
            if sum(y==1) != sum(y==0): #double check
                raise NameError('Oversampling failed!')
            else:
                print('Data has successfully been oversampled!')
                return X,y
        

"""
Load Data
"""

os.getcwd()
os.chdir('C:\\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Analysis\Data2')
file                      = os.listdir()


# Get indexes for the correct files
word = "MLdat1.mat"
ind = np.zeros(len(file))
for i in range(0,len(file)):
    strin = file[i]
    if re.search(word,file[i]) != None:
        ind[i] = 1
  

for part in range(0,len(file)):
    if ind[part] != 1:
        pass
    else:
        p_dat                     = loadmat(file[part])
        p_dat.keys() # get keys for the imported dictionary data
        print('Load data of participant %s' % part)
        
        
        
        # Location to save my output
        save_p = os.getcwd()[:-5]+'Results2\\' + file[part][:-10] +'_results.csv'
        cnt = 0
        
        #don't overwrite ever
        while os.path.exists(save_p):
            cnt += 1
        
            save_p = os.getcwd()[:-5]+'Results2\\' + file[part][:-10]  + '_v' + str(cnt) +'_results.csv'
    	
        
        
        type(p_dat['labels']),p_dat['labels'].shape # data type and data size - one dim
        type(p_dat['trials']),p_dat['trials'].shape # data type and data size - three dim
        
        
        # labels bring labels in correct format - row one long row vector
        labels                    = np.squeeze(np.transpose(p_dat['labels'])) - 1
        # For the data, rows = examples, columns = features 
        trl_dat                   = p_dat['trials']
        
        trl_dat, labels = oversampling(trl_dat, labels , part)
 
        """
       define the parameter space that will be searched over
        """
        
        C_range = np.logspace(-2, 10, 12, base =2)
        gamma_range = np.logspace(-9, 3, 12, base =2)
        alphas = np.logspace(-6, 6, 12)
        param_grid = dict(gamma=gamma_range, C=C_range)
        param_grid1 = dict(alpha= alphas) #params for ridge
        
        '''
        initiate loop of time bins
        '''
        for kernel in ('linear', 'rbf'):
               
            '''
            Create Data array to safe variables
            '''
            
            my_df                     = pd.DataFrame(data=np.array(range(1,trl_dat.shape[2]+1)), 
                                                     index=range(0,trl_dat.shape[2]), 
                                                     columns=['Bin'])
            my_df['interval']         = np.zeros(trl_dat.shape[2])
            my_df['accuracy']         = np.zeros(trl_dat.shape[2])
            my_df['specificity']      = np.zeros(trl_dat.shape[2])
            my_df['sensitivity']      = np.zeros(trl_dat.shape[2])
            my_df['precision']        = np.zeros(trl_dat.shape[2])
            my_df['F1']               = np.zeros(trl_dat.shape[2])
            my_df['gamma']            = np.zeros(trl_dat.shape[2])
            my_df['C']                = np.zeros(trl_dat.shape[2])
            my_df['alpha']            = np.zeros(trl_dat.shape[2])
            my_df['v_accuracy']       = np.zeros(trl_dat.shape[2])
             
            print('Performing SVM analysis with a %s kernel.' % kernel)
            # Location to save my output
            if kernel == 'linear':
                save_p = os.getcwd()[:-5]+'Results2\\' + file[part][:-10] +'_linR_results.csv'
            else:
                save_p = os.getcwd()[:-5]+'Results2\\' + file[part][:-10] +'_rbfK_results.csv'
            
            
            cnt = 0
            
            #don't overwrite ever
            while os.path.exists(save_p):
                cnt += 1
                if kernel == 'linear':
                    save_p = os.getcwd()[:-5]+'Results2\\' + file[part][:-10] + '_v' + str(cnt) + '_linR_results.csv'
                else:
                    save_p = os.getcwd()[:-5]+'Results2\\' + file[part][:-10] + '_v' + str(cnt) + '_rbfK_results.csv'
            
    	
            for t_bin in range(0,trl_dat.shape[2]): # time bins we are going to analyse
                data = trl_dat[:,:,t_bin]
                print('Analyzing data between {0} and {1} milliseconds.'.format(-20 +40 *t_bin -20*t_bin,
                      20 +40 *t_bin -20*t_bin)) #has to be adjusted
                
                
                
                #create stratified training and test splits 
                # NOTE: RANDOM STATE IS FIXED FOR EVERY PARTICIPANT 
                X_train, X_test, y_train, y_test = train_test_split(data, labels, 
                                                                    test_size=0.2, 
                                                                    stratify=labels, 
                                                                    random_state=part)
                
                
                
                 # now create a searchCV object and fit it to the data
                cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=part*7)
        #        loo_cv = LeaveOneOut()
                if kernel == 'rbf':
                    grid = GridSearchCV(svm.SVC(kernel=kernel, class_weight='balanced'), 
                                        param_grid=param_grid, cv=cv, refit = True)
            #        grid = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), 
            #                           param_grid=param_grid, cv=loo_cv, refit = True)
                else:
                    grid = RidgeClassifierCV(alphas = alphas,cv=cv, class_weight='balanced')
                                        
                 
                grid.fit(X_train, y_train)
                
                
#                
#                # Save parameter from CV
                if kernel == 'rbf':
                    print("The best parameters are %s with a score of %0.2f"
                      % (grid.best_params_, grid.best_score_))

                    my_df.loc[t_bin,'gamma'] =  round(grid.best_params_['gamma'],6)
                    my_df.loc[t_bin,'C'] =  round(grid.best_params_['C'],4)
                    my_df.loc[t_bin,'v_accuracy'] = round(grid.best_score_,4)
                else:
                    my_df.loc[t_bin,'alpha'] =  round(grid.alpha_,4)
                
                
              
#                # Test performance on the test set and save output
                performance = perf_measure(grid.predict(X_test), y_test)
                my_df.loc[t_bin, ['accuracy', 'sensitivity', 'specificity', 'precision', 'F1']] = performance
                
#                """
#                Note for the grid-predict scoring. Goes to show that just making sure that all
#                of our classes are equally represented in the training and valdiation samples is
#                not a good strategy. Results indicate that we build a classifier that simply 
#                rates all available examples as the majority class and by that achieves a higher
#                classification accuracy. Strategy I intend to follow instead is undersampling 
#                with a customade script. NOTE: setting classweight to 'balanced' did do no good 
#                either, same classifier behavior.
#                """
#               
                my_df.to_csv(save_p)