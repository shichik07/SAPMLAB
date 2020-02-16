# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:26:06 2020

@author: juliu
"""

"""
Plotting the results properly again. We will make two ERP plots of the CZ and Pz 
elctrode. One for both ERP conditions, and one for the difference waves. Both 
plots will include the within-subject confidence intervals after the Cousineau-
Morey Method: http://www.tqmp.org/Content/vol04-2/p061/p061.pdf

"""

from scipy.io import loadmat
import os
import numpy as np
import pandas as pd

os.getcwd()
os.chdir('C:\\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Analysis\Data')
file                      = os.listdir()
part = 1
p_dat                     = loadmat(file[part])
p_dat.keys() # get keys for the imported dictionary data
print('Load data of participant %s' % part)

# labels bring labels in correct format - row one long row vector
labels                    = np.squeeze(np.transpose(p_dat['labels'])) - 1
# For the data, rows = examples, columns = features 
trl_dat                   = p_dat['trials']

data_1 = trl_dat[1:2,:,1]
data_2 = trl_dat[3:4,:,1]
data = np.append(data_1,data_2,0)


def normalize_within(data):
    mean = np.mean(np.append(data[0,:],data[1,:]))
    print(data[1,44])
    data = data - mean
    #normalize with participant data by condition m_1 - (m_1+m_2)/2
    
def within_CI(data, alpha, C_nr, part_nr):
    for part in range(0,part_nr):
        data[:,:,part] = normalize_within(data[:,:,part])
    CI = np.zeros((2,data.shape[1]))
    for t_bins in range(0,data.shape[1]):
        bin_data = np.reshape(data[:,t_bins,:],(bin_data.shape[0]*bin_data.shape[1]))
        sd = np.std(bin_data)
        mu = np.mean(bin_data)
        CI[0,t_bins] = mu + 
    CI[0] = np.mean
    # CI = mean +- t(1-alpha/2*|n-1) * SD/sqrt(n)*sqrt(c/c-1) - Morreys correction
    return CIs
    

import seaborn as sns
sns.set()

fmri = sns.load_dataset("fmri")
sns.relplot(x="timepoint", y="signal", col="region",
            hue="event", style="event",
            kind="line", data=fmri);
            
print(fmri)

open("sns.relplot","r")
