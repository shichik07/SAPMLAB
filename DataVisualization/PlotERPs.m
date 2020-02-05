%% Plotting the ERP Data
clear all; close all; clc
dbstop if error
dirs.home                   = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Raw data\Processed';
ftp                         = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\EEG Labor\EEG Software\fieldtrip';
addpath(ftp)
cd(dirs.home);

ft_defaults;

% get directories
filenameFinal                = dir(['*_final.mat']);
filenameFinal                = {filenameFinal.name};
par.nSub                    = length(filenameFinal);

%% Plot ERPs
for iSub = 1:par.nSub;
end