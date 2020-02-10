%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                %
% EEG ML Preprocessing                           %                               
% Julius Kricheldorff(julius.kricheldorff@uol.de)%
%                                                %
%                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cut out only relevant data, from 0 to one seconds and put it into a
% three-dimensional array (?). Note the full analysis is to be performed in
% python, so the data format has to be appropriate.

clear all; close all; clc
dbstop if error
dirs.home                   = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis';
dirs.cur                    = fullfile([dirs.home, '\Raw data\Processed']);
dirs.save                   = fullfile([dirs.home, '\Analysis\Data\']);
cd(dirs.cur);


ftp                         = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\EEG Labor\EEG Software\fieldtrip';
addpath(ftp);
addpath(dirs.home); 
ft_defaults
filenameFinal                = dir(['*_final.mat']);
filenameFinal                = {filenameFinal.name};
par.nSub                     = length(filenameFinal);


% Standardize data - rationale being that we only want the data normalized
% with respect to relevant activity

for iSub                = 1:par.nSub;

    load(filenameFinal{iSub});
    % provide information on screen.
    sprintf('Plotting data loaded of sID = %d',par.sID)
    
    % indicate save location
    subTag                  = filenameFinal{iSub}(1:end-10);
    filenameProc = fullfile([dirs.save, subTag, '_MLdat.mat']);
    
    % Just use the relevant samples for the analysis. Note: I retain 0.2 ms
    % prior to the stimulus presentation. The ERP plots made me suspicious
    % that the period may already contain classification relevant
    % information. I want to use the classification performance to either
    % discern or confirm my suspicions. Moreover at this point I will not
    % perform any hp filtering
    % select indexes for correctly answered Go and NoGo trials
    NoGoCorrect         = [seq.EEG.accuracy == 1 & seq.EEG.resp ==0];
    GoCorrect           = [seq.EEG.accuracy == 1 & [seq.EEG.resp ==97 |...
        seq.EEG.resp ==101]]; %Left and Right responses
    
    Trl_indices = logical([NoGoCorrect +  GoCorrect]);
    
    cfg                     = [];
    fg.preproc.lpfilter     = 'yes';
    cfg.latency             = [-0.2 1]; 
    cfg.trials              = Trl_indices; 
    cfg.keeptrials          = 'yes'; % yes we want to have the single trial data
    cfg.removemean          = 'no'; % i want to perform the standardization myself
    data                    = ft_timelockanalysis(cfg,data);
    
    data.labels             = [NoGoCorrect +  GoCorrect*2]; % 2 for Go, 1 for NoGo
    labels             = data.labels(Trl_indices);
    
    % Perform standardization for every channel and every trial separately
    
    m_dat                   = mean(data.trial, 3);
    sd_dat                  = std(data.trial, 0, 3); % 0 indicating n-1 df
    
    data.trial              = (data.trial - m_dat)./sd_dat;
    
    % Cut data via sliding window into 40 ms bins with 20ms overlap in order
    % not to miss any relevant periods (initially wanted to do 50 and 25,
    % however that doesnt work with 500 Hz ;p
    trials              = mean_sliding_window(40, 20, data.trial, 500);
    
    save(filenameProc, 'trials', 'labels');
end
