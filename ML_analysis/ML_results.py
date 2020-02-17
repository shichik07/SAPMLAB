# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 20:26:06 2020

@author: juliu
"""

"""
Plotting the results properly again. We will make two ERP plots of the CZ and Pz 
elctrode. One for both ERP conditions, and one for the difference waves. Both 
plots will include the within-subject confidence intervals after the Cousineau-
Morey Method: http://www.tqmp.org/Content/vol04-2/p061/p061.pdf. A cautionary 
note in terms of usage. For my own project I deemed it not sufficiently relevant,
but for the CI calculation I did not use a weighted mean. However, some of the 
avarage calculations for each participant are based on more or fewer samples. So
for an accurate calculations it d probably d be wise to use the weighted SD and 
mean.

"""

from scipy.io import loadmat
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
from scipy.stats import t
import re 

os.getcwd()
os.chdir('C:\\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Analysis\Data')
file                      = os.listdir()


# Get indexes for the correct files
word = "ERP_data_all.mat"
ind = 0
for i in range(0,len(file)):
    strin = file[i]
    if re.search(word,file[i]) != None:
        ind = i
   
# load data

p_dat                     = loadmat(file[ind])
p_dat.keys() # get keys for the imported dictionary data

# For the data, rows = examples, columns = features 
Go_data                     = p_dat['All_dataGo']
NoGo_data                   = p_dat['All_dataNoGo']
time             = np.squeeze(p_dat['time'])


def normalize_within(data1, data2):
    #normalize with participant data by condition m_1 - (m_1+m_2)/2
    mean = np.mean(np.append(data1, data2, axis = 1), axis = 1)
    data1[0,:] = data1[0,:] - mean[0]
    data1[1,:] = data1[1,:] - mean[1]
    data2[0,:] = data2[0,:] - mean[0]
    data2[1,:] = data2[1,:] - mean[1]
   # shitty way to do it, but apprently np is not the same as matlab and i am too
   #tired to figure it out better now
    return data1, data2



def calculate_CI(data, alpha, C_nr, part_nr):
    CI = np.zeros((4,data.shape[1]))
    t_value = t.ppf(1-(alpha/2),part_nr-1)
    for t_bins in range(0,data.shape[1]):
        bin_data = data[:,t_bins,:]
        sd_Cz = np.std(bin_data[0,:])
        sd_Pz = np.std(bin_data[1,:])
        mu_Cz = np.mean(bin_data[0,:])
        mu_Pz = np.mean(bin_data[1,:])
        # CI = mean +- t(1-alpha/2*|n-1) * SD/sqrt(n)*sqrt(c/c-1) - Morreys correction
        CI[0,t_bins] = mu_Cz + t_value*(sd_Cz/np.sqrt(part_nr)) *np.sqrt(C_nr/(C_nr-1))
        CI[1,t_bins] = mu_Cz - t_value*(sd_Cz/np.sqrt(part_nr)) *np.sqrt(C_nr/(C_nr-1))
        CI[2,t_bins] = mu_Pz + t_value*(sd_Pz/np.sqrt(part_nr)) *np.sqrt(C_nr/(C_nr-1))
        CI[3,t_bins] = mu_Pz - t_value*(sd_Pz/np.sqrt(part_nr)) *np.sqrt(C_nr/(C_nr-1))
    return CI
    

def within_CI(dGo, dNoGo, alpha, C_nr, part_nr):
    # calculate CIs on normalized data and correct for nr of comparisons
    for part in range(0,part_nr):
        dGo[:,:,part], dNoGo[:,:,part] = normalize_within(dGo[:,:,part], dNoGo[:,:,part])
    CIGo = calculate_CI(dGo, alpha, C_nr, part_nr)
    CINoGo = calculate_CI(dNoGo, alpha, C_nr, part_nr)
    grandAverageGo = np.mean(dGo, axis = 2)
    grandAverageNoGo = np.mean(dNoGo, axis = 2)
    return CIGo, grandAverageGo,  CINoGo, grandAverageNoGo
    

CI_go, avg_go, CI_nogo, avg_nogo = within_CI(Go_data, NoGo_data, 0.05, 2, 5)


"""
Finally lets start plotting
Code is Courtesy of Travis DeWolf:    
https://studywolf.wordpress.com/2017/11/21/matplotlib-legends-for-mean-and-confidence-interval-plots/ 
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
 
def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)
    plt.ylabel('μV')
    plt.xlabel('time in ms')
    time = np.array([-200,0,200,400,600,800,1000])
    pos = np.array([0,100,200,300,400,500,600])
    plt.xticks(pos, time)
    
class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
 
        return patch

fig = plt.figure(1, figsize=(8, 4.5))
plot_mean_and_CI(avg_go[0,:], CI_go[1,:], CI_go[0,:], color_mean='k--', color_shading='k')
plot_mean_and_CI(avg_nogo[0,:], CI_nogo[1,:], CI_nogo[0,:], color_mean='b', color_shading='b')

bg = np.array([1, 1, 1])  # background of the legend is white
colors = ['black', 'blue', 'green']
# with alpha = .5, the faded color is the average of the background and color
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
 
plt.legend([0, 1], ['GoData', 'NoGoData'],
           handler_map={
               0: LegendObject(colors[0], colors_faded[0], dashed=True),
               1: LegendObject(colors[1], colors_faded[1]),
            })
 
plt.title('ERP at Cz Electrode')
plt.tight_layout()
plt.grid()
plt.show()
#plt.savefig(fname = "CzElectrode.png", dpi=150)

# Plot the second figure

fig = plt.figure(2, figsize=(8, 4.5))
plot_mean_and_CI(avg_go[1,:], CI_go[3,:], CI_go[2,:], color_mean='b--', color_shading='b')
plot_mean_and_CI(avg_nogo[1,:], CI_nogo[3,:], CI_nogo[2,:], color_mean='darkslategray', color_shading='g')


bg = np.array([1, 1, 1])  # background of the legend is white
colors = ['black', 'blue', 'green']
# with alpha = .5, the faded color is the average of the background and color
colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
 git 
plt.legend([0, 1], ['GoData', 'NoGoData'],
           handler_map={
               0: LegendObject(colors[1], colors_faded[1],dashed=True),
               1: LegendObject(colors[0], colors_faded[2]),
            })
 
plt.title('ERP at Pz Electrode')
plt.tight_layout()
plt.grid()
plt.show()
