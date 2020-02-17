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
par.nSub                     = length(filenameFinal);


% Files for Grandaverage
g_avg = {};

%% Plot ERPs for single participants
for iSub                = 1:par.nSub;

    load(filenameFinal{iSub});
    
    % provide information on screen.
    sprintf('Plotting data loaded of sID = %d',par.sID)
    
    % select indexes for correctly answered Go and NoGo trials
    
    GoCorrect           = [seq.EEG.accuracy == 1 & [seq.EEG.resp ==97 | seq.EEG.resp ==101]];
    NoGoCorrect         = [seq.EEG.accuracy == 1 & seq.EEG.resp ==0];
    GoAll               = [seq.EEG.resp ==97 | seq.EEG.resp ==101]; %Left and Right responses
    
%     
%     % remove the newly marked epochs from data.
%     cfg                     = [];
%     cfg.trials              = 'all';
%     cfg.preproc.lpfilter        = 'yes';
%     cfg.preproc.lpfreq     = 35;
%     data                   = ft_preprocessing(cfg,data);
%     
    % Data GoCorrect 
    cfg                 = [];
    cfg.preproc.lpfilter        = 'yes';
    cfg.preproc.lpfreq          = 35;
    cfg.channel         = 'all';
    cfg.trials          = GoCorrect;
    cfg.keeptrials      = 'no'; % yes if necesaary for the statistics to work
    timelockGoCor       = ft_timelockanalysis(cfg,data);
    
    % Data GoIncorrect
    cfg                 = [];
    cfg.preproc.lpfilter        = 'yes';
    cfg.preproc.lpfreq          = 35;
    cfg.channel         = 'all';
    cfg.trials          = GoCorrect;
    cfg.keeptrials      = 'no'; % necesaary for the statistics to work
    timelockGoInco      = ft_timelockanalysis(cfg,data);
    
    % Data NoGoCorrect
    cfg                 = [];
    cfg.preproc.lpfilter        = 'yes';
    cfg.preproc.lpfreq          = 35;
    cfg.channel         = 'all';
    cfg.trials          = NoGoCorrect;
    cfg.keeptrials      = 'no'; % necesaary for the statistics to work
    timelockNoGo        = ft_timelockanalysis(cfg,data);
    
    % Plots for a single channel --> ft_singleplotER() at Fz for the
    % frontal P3 - Fz = 1, Pz = 5
    
    cfg                 = [];
    cfg.xlim            = [-0.5 1.0]; %indicate time window
    cfg.ylim            = [-8 8];
    
    cfg.channel         = '5'; % 1 corresponds to Cz electrode, 5 to Pz electrode
    cfg.fontsize        = 18;
    cfg.linewidth       = 2;
    figure; ft_singleplotER(cfg,timelockNoGo, timelockGoCor);
    yline(0)
    xline(0)
    legend('NoGO','Go')
    
    % Multiplot
    cfg.channel = 'all';
    cfg.layout = par.layout; 
    ft_multiplotER(cfg,timelockNoGo, timelockGoCor);
    
    %NOTE: SIngle Subject ERPs do not look very convincing. I am not sure
    %whether or not something went wrong during preprocessing, but there
    %appear to be very large differences in the single subject ERPs. What
    %is particularly troublesome however, is that the P3 ERP, which should
    %start peakin between 300-600ms after stimulus onset, peaks already
    %quite early ~290ms. Is my EEG dataset shifted to the right? Baseline
    %looks also a bit sketchy for some participants.
    
    g_avg.GoCor{iSub} = timelockGoCor;
    g_avg.GoInco{iSub} = timelockGoInco ;
    g_avg.NoGO{iSub} = timelockNoGo ;
    
end

%% Plotting the grandaveraged data 

% Calculate grandaverages

cfg                     = [];
cfg.method              = 'within'; %weighted average
g_avg.all.GoCor         = ft_timelockgrandaverage(cfg, g_avg.GoCor{1,1}, g_avg.GoCor{1,2},...
    g_avg.GoCor{1,3},g_avg.GoCor{1,4},g_avg.GoCor{1,5});

cfg                     = [];
cfg.method              = 'within'; %weighted average
g_avg.all.GoInco        = ft_timelockgrandaverage(cfg, g_avg.GoInco{1}, g_avg.GoInco{2},...
    g_avg.GoInco{3}, g_avg.GoInco{4}, g_avg.GoInco{5});

cfg                     = [];
cfg.method              = 'within'; %weighted average
g_avg.all.NoGo          = ft_timelockgrandaverage(cfg, g_avg.NoGO{1}, g_avg.NoGO{2},...
    g_avg.NoGO{3}, g_avg.NoGO{4}, g_avg.NoGO{5});


% Plot grandaverages
cfg                 = [];
cfg.xlim            = [-0.5 1.0]; %indicate time window
cfg.ylim            = [-8 8];
cfg.channel         = {'1', '5'}; % 1 corresponds to Cz electrode, 5 to Pz electrode
cfg.fontsize        = 18;
cfg.linewidth       = 2;
figure; ft_singleplotER(cfg, g_avg.all.NoGo, g_avg.all.GoCor);
yline(0)
xline(0)
legend('NoGO','Go')

% Plot the difference wave 

g_avg.all.diff      = g_avg.all.NoGo;
g_avg.all.diff.avg  = g_avg.all.NoGo.avg - g_avg.all.GoInco.avg;


cfg                 = [];
cfg.xlim            = [-0.5 1.0]; %indicate time window
cfg.ylim            = [-8 8];
cfg.channel         = {'1', '5'}; % 1 corresponds to Cz electrode, 5 to Pz electrode
cfg.fontsize        = 18;
cfg.linewidth       = 2;
figure; ft_singleplotER(cfg, g_avg.all.diff);
yline(0)
xline(0)
legend('Difference Wave')

%% Summary
% To summarize my plotting results, I am slightly concerned about the data.
% Granted it is very few participants, but from the averaged ERP waveforms
% it appears that the decision process (if you will) already starts prior
% to the stimulus-lock as indicated by the decrease. Moreover, overall
% differences between the two conditions seem relatively small, as
% indicated by the difference wave. Based on the ERP data alone I would be
% very uncertain with regards to any conclusion.


%% Save ERP data for Cz and Pz electrodes
cfg.channel         = {'1', '5'};

All_dataGo              = zeros(2,3000,5);
All_dataNoGo            = zeros(2,3000,5);

for iSub                = 1:5;
    load(filenameFinal{iSub});
    % provide information on screen.
    sprintf('Plotting data loaded of sID = %d',par.sID)
    
    % select indexes for correctly answered Go and NoGo trials
    
    GoCorrect           = [seq.EEG.accuracy == 1 & [seq.EEG.resp ==97 | seq.EEG.resp ==101]];
    NoGoCorrect         = [seq.EEG.accuracy == 1 & seq.EEG.resp ==0];
    GoAll               = [seq.EEG.resp ==97 | seq.EEG.resp ==101]; %Left and Right responses
 
    % Data GoCorrect 
    cfg                 = [];
    cfg.preproc.lpfilter        = 'yes';
    cfg.preproc.lpfreq          = 35;
    cfg.channel         = {'1', '5'};
    cfg.trials          = GoCorrect;
    cfg.keeptrials      = 'no'; % yes if necesaary for the statistics to work
    timelockGoCor       = ft_timelockanalysis(cfg,data);
    
    % Data GoIncorrect
    cfg                 = [];
    cfg.preproc.lpfilter        = 'yes';
    cfg.preproc.lpfreq          = 35;
    cfg.channel         = {'1', '5'};
    cfg.trials          = GoCorrect;
    cfg.keeptrials      = 'no'; % necesaary for the statistics to work
    timelockGoInco      = ft_timelockanalysis(cfg,data);
    
    % Data NoGoCorrect
    cfg                 = [];
    cfg.preproc.lpfilter        = 'yes';
    cfg.preproc.lpfreq          = 35;
    cfg.channel         = {'1', '5'};
    cfg.trials          = NoGoCorrect;
    cfg.keeptrials      = 'no'; % necesaary for the statistics to work
    timelockNoGo        = ft_timelockanalysis(cfg,data);
    
    % Plots for a single channel --> ft_singleplotER() at Fz for the
    % frontal P3 - Fz = 1, Pz = 5
  
    
    time                = timelockGoCor.time;
    a = timelockGoCor.avg;
    All_dataGo(1:2,:,iSub)    = timelockGoCor.avg;
    All_dataNoGo(:,:,iSub)  = timelockNoGo.avg;
end

dirs.ML = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\Analysis\Data';
save_pl = fullfile(dirs.ML,'ERP_data_all.mat');
save(save_pl, 'All_dataGo','All_dataNoGo', 'time')