%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                %
% Data preprocessing                             %                               
% Julius Kricheldorff(julius.kricheldorff@uol.de)%
%                                                %
%                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% DISCLAIMER: THe Script below is an adjusted version of the script provided by the
% authors of the original study/dataset. FOr reading in the data, I only
% changed High-passfilter (1Hz instead of 0.5) and otherwise left all
% settings for importing the data (includes referencing, re-referncing
% steps. All steps beyond importing the data are my own work, i.e. based on
% tutorial scripts available on the fieldtrip webiste. That includes
% artifact rejection (manually), rejection of bad trials, frequency
% transformation (if I manage to get there in time). For a preliminary
% analysis I also only used 5 instead of the
% 

%%

clear all; close all; clc
dbstop if error
ftp                         = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\EEG Labor\EEG Software\fieldtrip';
% set directories
dirs.home                   = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\';

% add fieldtrip and set defaults
addpath (ftp);
ft_defaults;

% get subject directories
dirs.log                    = fullfile(dirs.home,'Raw data');
cd(dirs.log)
sublist                     = dir(dirs.log);
sublist                     = {sublist.name};
par.nSub                    = length(sublist)-2; % - 2 because the first two indexes contain nothing and I am not sure why

%% epoch information
% In terms of event codes, I was able to extract the following from the
% experimental file: 

% The stimulus sequence of 'Go-to-win-left'(1),'Go-to-win-right'(2),'Go-to-avoid-left'(3),
% 'Go-to-avoid-right'(4),'NoGo-to-win'(5),'NoGo-to-win'(6),'NoGo-to-avoid'(7),'NoGo-to-avoid'(8),
% and feedback validity depending on specified probability. mixed in blocks.
% -------------------------------------------------------------------------

% What is a bot confusing to me still is the total number of trials. While
% in the publication 320 trials are named as the accurate number, in the
% data set we find actually a total of 640 trials. Perhaps a different
% publication? On the other hand, I only get behavioral feedback for 320
% trials

% UPDATE: Okay from what I could gather looking through the publication:

%"Frontal network dynamics reflect neurocomputational mechanisms for 
% reducing maladaptive biases in motivated action"

% They actually let their participants perform the task twice (320 trials
% each) with two independent sets of stimuli. For the analysis in the
% publication the apparently however, only used one of both tasks. Not sure
% why. The behavioral data is however available in the corresponding
% results.mat file. Hence I can feel free to use both datasets for my
% ML-Analysis.
for iCue = 1:8
    par.eventCode{iCue}     = sprintf('S%d',110+iCue); % s in ft
end
par.epochtime               = [-1.5 4.5];                  % about 1 sec before & after trial

% information for artifact rejection.
par.time4TR                 = [-1.25 4]; % critial time window for artifact rejection.

% high pass filter in Hz.
par.hpf                     = 1; % As mentioned I am removing frequency lower than 1Hz

% linear baseline correction.
par.base4correction         = [-.2 0]; % baseline correction seems to be a sensible choice for me 

% channel information.
par.chan.VEOG               = {'61' '62'};
par.chan.HEOG               = {'63' '64'};
par.chan.REF                = {'REF' '53'};
par.chan.EEG                = [1:60 65];
par.chan.layoutfile         = 'dccn_customized_acticap64'; % 2D layout
par.chan.positionfile       = 'easycap-M10.txt';         % 3D coordinates

% check layout.
cfg                         = [];
cfg.layout                  = par.chan.layoutfile;
par.layout                  = ft_prepare_layout(cfg);
figure
ft_plot_lay(par.layout);
title(cfg.layout);

% check electrode positions.
channelpositions            = ft_read_sens(par.chan.positionfile);
figure; ft_plot_sens(channelpositions,'label','label');
title('channel positions');
% the ref position is ignored and should be added between
% otherwise electrode positions will be inaccurate
channelpositions.label{59}  = 'REF';    % redefine REF as 59
channelpositions.label{60}  = '59';     % move 59 to 60
channelpositions.label{61}  = '60';     % move 60 to 61
par.chanpos                 = channelpositions;
figure; ft_plot_sens(channelpositions,'label','label');
title('channel positions');
clear channelpositions


%% loop over subjects.
for iSub = 1:par.nSub;
    iSub = iSub + 2; % Same goes here to indicate the correct index
    
    % get subject specific directory. EDIT: Julius - I also added the
    % header file manually here, otherwise reading in the data did not work
    % for me.
    dirs.rawData            = fullfile(dirs.log, sublist{iSub});
    cd(dirs.rawData)
    dirs.fileArray          = dir('*.eeg');
    dirs.fileArray          = dirs.fileArray.name;
    dirs.HeaderArray          = dir('*.vhdr');
    dirs.HeaderArray          = dirs.HeaderArray.name;

    
    
    sID                     = str2double(dirs.fileArray(end-5:end-4)); % s30 named differently during recording (old: sID = dirs.fileArray(8:11); )
    subTag                  = dirs.fileArray(1:end-4);
    filenamePreTR           = fullfile(dirs.log,'preTR',[subTag '_preTR.mat']);
    
    % skip subjects when preprocessing is already done.
    if exist(filenamePreTR,'file')
        continue;
    end
    
    % provide information on screen.
    sprintf('Preprocessing sID = %d',sID);
    
    % store directories and subject ID.
    par.dirs                = dirs;
    par.sID                 = sID;
    
    % load behavioural data.
    resultsfile             = dir('*_results.mat');
    load(resultsfile.name)
    
    % epoch/trigger information.
    par.nCue                = prep.par.nStim;
    par.code                = prep.par.code; % triggercodes (s before number in ft)
    par.condiID             = prep.par.stimID;
    for iCue = 1:par.nCue
        par.eventCode{iCue} = sprintf('S%d',par.code.startTrial(iCue)); % s in ft
    end
    for iCue = 1:par.nCue
        par.eventCode{iCue} = sprintf('S%d',110+iCue); % s in ft
    end
    
    % define epochs.
    cfg                     = [];
    cfg.dataset             = fullfile(dirs.rawData,dirs.fileArray);
    cfg.trialdef            = [];
    cfg.trialdef.eventvalue = par.eventCode;
    cfg.trialdef.eventtype  = 'Stimulus';
    cfg.trialdef.prestim    = abs(par.epochtime(1));
    cfg.trialdef.poststim   = abs(par.epochtime(2));
    cfg.event               = ft_read_event(dirs.HeaderArray);
    cfg                     = ft_definetrial(cfg);
    
    % read in epochs.
    data                    = ft_preprocessing(cfg);
    
    
    %% Swap channels back when electrodes were switched during recording.
    if sID == 14;
        for iTrial = 1:size(data.trial,2);
            data.trial{iTrial}([54 4],:) = data.trial{iTrial}([4 54],:);
        end
    elseif sID == 32;
        for iTrial = 1:size(data.trial,2);
            data.trial{iTrial}([54 34],:) = data.trial{iTrial}([34 54],:);
        end
    elseif sID == 33;
        for iTrial = 1:size(data.trial,2);
            data.trial{iTrial}([54 34],:) = data.trial{iTrial}([34 54],:);
        end
    end
    
    
    %% remove redundant triggers.
    % NOTE: some subjects had one or two additional triggers. Remove cue
    % trigger that follows previous cue trigger within 1300ms (cue 
    % presentation is 1300 ms) and check if trial sequence from behavioural
    % and EEG file than correspond.
    
    if  size(data.trial,2) > 640
        
        % print subject ID with number of trials.
        sprintf('\n\n%d had %d trials!\n\n',sID,numel(data.trial));
        
        % retrieve trialsequence from behavioural and EEG file.
        stim = [prep.seq.learn.stim{1};prep.seq.learn.stim{2}];
        EEGstim = data.trialinfo-110;
        
        % sID had one trigger in the middle of a trial. remove that trigger
        % here, to make the procedure work.
        if sID == 43; 
            cfg                 = [];
            cfg.trials          = 1:size(data.trial,2);
            cfg.trials(500)     = [];
            data                = ft_selectdata(cfg, data);
            EEGstim = data.trialinfo-110;
        end
        
        % compare trialsequences to see if this approach works correctly.
        EEGstim([0; diff(data.sampleinfo(:,1))<1300]==1) = [];
        if any(EEGstim-stim)
            error('check triggers carefully!')
        end
        
        % remove the double trigger.
        cfg                 = [];
        cfg.trials          = ~[0; diff(data.sampleinfo(:,1))<1300];
        data                = ft_selectdata(cfg, data);
    end
    
    
    %% Complete preprocessing up to trial rejection.
    
    % rereferencing EEG channels.
    % step 1 - completely to 2nd reference electrode.
    cfg                     = [];
    cfg.reref               = 'yes';
    cfg.refchannel          = par.chan.REF{2};
    cfg.implicitref         = par.chan.REF{1};
    data                    = ft_preprocessing(cfg, data);
    % step 2 - weighted to 1st and 2nd reference electrode (otherwise
    % skewed to the left REF).
    cfg                     = [];
    cfg.reref               = 'yes';
    cfg.refchannel          = par.chan.REF;
    data                    = ft_preprocessing(cfg,data);
    
    % rereferencing V/HEOG channels to each other.
    % VEOG.
    cfg                     = [];
    cfg.reref               = 'yes';
    cfg.channel             = par.chan.VEOG;
    cfg.refchannel          = par.chan.VEOG{2};
    VEOG                    = ft_preprocessing(cfg,data);
    cfg                     = [];
    cfg.channel             = par.chan.VEOG{1};
    VEOG                    = ft_selectdata(cfg,VEOG);
    VEOG.label              = {'VEOG'};
    % HEOG.
    cfg                     = [];
    cfg.reref               = 'yes';
    cfg.channel             = par.chan.HEOG;
    cfg.refchannel          = par.chan.HEOG{2};
    HEOG                    = ft_preprocessing(cfg,data);
    cfg                     = [];
    cfg.channel             = par.chan.HEOG{1};
    HEOG                    = ft_selectdata(cfg,HEOG);
    HEOG.label              = {'HEOG'};
    % merge with EEG data.
    cfg                     = [];
    cfg.channel             = {data.label{par.chan.EEG}};
    data                    = ft_selectdata(cfg, data);
    cfg                     = [];
    data                    = ft_appenddata(cfg, data, VEOG, HEOG);
    par.chan.EEG            = 1:size(data.trial{1},1) - 2;
    clear VEOG HEOG
    
    % high-pass filter.
    cfg                     = [];
    cfg.hpfilter            = 'yes';
    cfg.hpfreq              = par.hpf;
    data                    = ft_preprocessing(cfg,data);
    
    % linear baseline correction.
    cfg                     = [];
    cfg.demean              = 'yes';
    cfg.baselinewindow      = par.base4correction;
    data                    = ft_preprocessing(cfg,data);
    
    % update parameters.
    par.nTrial              = length(data.trial);
    par.nChan               = size(data.trial{1},1);
    par.nTime               = size(data.trial{1},2);
    
    % save preprocessed data.
    save(filenamePreTR,'data','par')
    
end % end iSub-loop

%% From hereon, I only 
dirs.log                    = fullfile(dirs.home,'Raw data','preTR');
cd(dirs.log);
filenamePreTR               = dir('*_preTR.mat');
filenamePreTR               = {filenamePreTR.name};
par.nSub                    = length(filenamePreTR);

%% Preprocessing I - noisy trial removal
for iSub = 1:par.nSub
    
    rejects = [filenamePreTR{iSub}(1:12), 'rejected_trls']; %create filename for trl reject data
    load(filenamePreTR{iSub},'data','par');
    
    
    % skip subjects when preprocessing is already done.
    if exist(filenamePreTR{iSub},'file') ~= 2
        continue;
    end
    
    
    cfg = [];
    cfg.viewmode                = 'vertical';
    cfg.allowoverlap            = 'yes';
    cfg.continuous              = 'no';
    cfg.plotlabels              = 'yes';
    cfg                         = ft_databrowser(cfg,data) 
    par.visualArtf              = cfg.artfctdef.visual.artifact
    
    % THe authors of the original study did absolute rubbish here.
    % Apparently fieldtrip does not allow the usage of the
    % ft_rejectartifact function when there is overlap between the trials
    % (which is the case due to an excessively long baseline period).
    % Hence, they exported the fieldtrip data into the eeglab toolbox, used
    % the databrowser to mark artifactual trials manually. EEGlab then
    % outputs a binary array of the selcted trials, which then again was
    % re-imported into fieldtrip, and used in the ft_selectdata function to
    % reject the artifactual trials chosen in eeglab. Not sure if this is
    % the most elegant solution. I for my part will try to implement all of
    % it simply in fieldtrip, With a function that outputs me a binary
    % array. Saves a ton of time and lines of code I'd think.
    functions = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis'
    addpath(functions)
    rejected_trials = get_trlrej(par.visualArtf, data.sampleinfo);
    save(rejects,'rejected_trials')
    
    % Note: participant 16 had extremely noisy data, over 160 trials had to
    % be removed and that was a conservative guess. Participant 15 was
    % similarly problematic. However, the problem was not excessive
    % movement, but drowsyness. The second part of the experiment was
    % cluttered with alpha oscillations. Removed ~70 trials, perhaps to
    % conservative as well.
end
%% Rejecting trials and ICA

dirs.log                    = fullfile(dirs.home,'Raw data','preTR');
cd(dirs.log);
filenamePreTR               = dir('*_preTR.mat');
filenamePreTR               = {filenamePreTR.name};
par.nSub                    = length(filenamePreTR);
dirs.log1                    = fullfile(dirs.home,'Raw data','ICA');


for iSub = 1:par.nSub
    rejects = [filenamePreTR{iSub}(1:12), 'rejected_trls']; %create filename for trl reject data
    load(filenamePreTR{iSub},'data','par');
    load(rejects,'rejected_trials');
    filenameICA = [dirs.log1, '_',filenamePreTR{iSub}(1:12)];
    
    % skip subjects when ICA is already done.
    if exist(filenamePreTR{iSub},'file') ~= 2
        continue;
    end
    
    % provide information on screen.
    sprintf('Preprocessing sID = %d',par.sID);
    
    
    
    % remove marked epochs from data.
    cfg                     = [];
    cfg.trials              = find(~rejected_trials);
    data                    = ft_selectdata(cfg,data);
    
    % update number of trials
    par.nTrial              = size(data.trial,2);
     
%     % remove HEOG before ICA
%     cfg                     = [];
%     cfg.channel             = {data.label{1:end-1}};
%     data                    = ft_selectdata(cfg, data);
%     par.nChan               = size(data.trial{1},1);
    
    % run ICA (runica is default).
    cfg                     = [];
    cfg.channel             = [1:60]; % don't include the eog channels
    comp                    = ft_componentanalysis(cfg, data);
    
    % save data before IC rejection
    save(filenameICA,'comp','par','rejected_trials');
    clear data
    
end % end iSub-loop. 
    


%% Preprocessing II - Identifying Independent Components
% I again used the script of the original authors. R

clear all; close all; clc
dbstop if error

% set directories.
dirs.home                   = '~';

% add fieldtrip and set defaults.
addpath /home/common/matlab/fieldtrip;
ft_defaults;

% get subject directories.
dirs.log                    = fullfile(dirs.home,'projects','EEG','Log','ICA');
cd(dirs.log);
filenameICs                 = dir('*_IC.mat');
filenameICs                 = {filenameICs.name};
par.nSub                    = length(filenameICs);


dirs.log                    = fullfile(dirs.home,'Raw data','ICA');
cd(dirs.log);
filenameICs                 = dir(['ICA_','*_.mat']);
filenameICs                 = {filenameICs.name};
par.nSub                    = length(filenameICs);


c = colormap(prism(640));
    

%% loop over subjects.
for iSub = 1:par.nSub;
    iSub = 5
    close all
    
    % display subject information on screen.
    sprintf(filenameICs{iSub})
    
    % load data from previous step
    load(filenameICs{iSub})
    
    % retrieve component activations.
    compActivation = cat(3,comp.trial{:});
    
    % retrieve frequency spectrum of components (adapted from ft_icabrowser.m). 
    % Takes a while -> minute or so for 61 components :)
    fft_data = cat(2,comp.trial{1:5:end});
    smo = 50;
    steps = 10;
    Fs = comp.fsample;
    N = floor(size(fft_data,2));
    
    for iComp = 1:numel(comp.label)
        xdft = fft(fft_data(iComp,:));
        xdft = xdft(1:N/2+1);
        psdx = (1/(Fs*N)).*abs(xdft).^2;
        psdx(2:end-1) = 2*psdx(2:end-1);
        
        j = 1; k = 1;
        while j < length(psdx)-smo
            smoothed{iComp}(k)=mean(psdx(j:j+smo));
            j = j + steps; k = k + 1;
        end
        
        freq{iComp} = linspace(0,Fs/2,size(smoothed{iComp},2));
        strt{iComp} = find(freq{iComp} > 2,1,'first');
        stp{iComp}  = find(freq{iComp} < 200,1,'last');
    end
    
    % View Topoplots of components per 20.
    figure('Position',[200 200 800 800]);
    cfg                     = [];
    cfg.layout              = par.chan.layoutfile;
    cfg.component           = 1:20;
    ft_topoplotIC(cfg,comp);
    saveas(gcf,[filenameICs{iSub}(1:end-4) '_1.jpg'])
    figure('Position',[200 200 800 800]);
    cfg.component           = 21:40;
    ft_topoplotIC(cfg,comp);
    saveas(gcf,[filenameICs{iSub}(1:end-4) '_2.jpg'])
    figure('Position',[200 200 800 800]);
    cfg.component           = 41:60;
    ft_topoplotIC(cfg,comp);
    saveas(gcf,[filenameICs{iSub}(1:end-4) '_3.jpg'])
    
    % plot for component 1-20 topoplot, trial-by-time activation and power
    % spectrum.
    for iComp = 1:numel(comp.label)
%     for iComp = 1
        figure('Position',[100 600 1800 300]);
        subplot(1,5,1)
        cfg                     = [];
        cfg.layout              = par.chan.layoutfile;
        cfg.component           = iComp;
        ft_topoplotIC(cfg,comp);
        % Trial-by-time activation.
        subplot(1,5,2)
        imagesc(comp.time{1},1:par.nTrial,squeeze(compActivation(iComp,:,:))')
        set(gca,'clim',[-.5*mean(abs(get(gca,'clim'))) .5*mean(abs(get(gca,'clim')))],...
            'ytick',0:100:par.nTrial,'ydir','normal')
        xlabel('Time (s)'); ylabel('Trial')
        % Event-related activation.
        subplot(1,5,3)
        plot(comp.time{1},squeeze(mean(compActivation(iComp,:,:),3)))
        set(gca,'xlim',comp.time{1}([1 end])); xlabel('Time (s)'); 
        % activation of selected component for all trials.
        subplot(1,5,4); hold on
        for iTrial = 1:size(compActivation,3)
            plot(comp.time{1},squeeze(compActivation(iComp,:,iTrial)),...
                'Color',c(iTrial,:))
        end
        set(gca,'xlim',comp.time{1}([1 end])); xlabel('Time (s)'); 
        % power spectrum.
        subplot(1,5,5)
        plot(freq{iComp}(strt{iComp}:stp{iComp}),...
            log10(smoothed{iComp}(strt{iComp}:stp{iComp})));
        set(gca,'TickDir','out','XTick',0:10:50,'xlim',[1 50])
        xlabel('Frequency (Hz)'); ylabel('(dB/Hz)');
    end
    
    keyboard
    
%     % view activation of all components trial-by-trial.
%     cfg                     = [];
%     cfg.viewmode            = 'vertical';
%     cfg.channel             = 'all';
%     ft_databrowser(cfg, comp);

end % end iSub-loop.

%% COMPONENTS identified for rejection
% I was very conservative here, better save than sorry
% Participant 16 - C1 - Eyeblink, C9 Noise, C30 Noise
% Participant 13 - C2 Eyebling, C11 - Noise, C23 Noise
% Participant 14 - C1 Eyeblink
% Participant 15 - C8 Eyeblink
% Participant 17 - C4 Eyeblink
%% Overview of components to reject 
%% Preprocessing III - LP Filtering

clear all; close all; clc
dbstop if error
ft_defaults;

comp2remove                 = { % sID, [IC to remove]
    16, [1 9 30]; % C1 - Eyeblink, C9 Noise, C30 Noise
    13, [2 11 23]; % C2 Eyebling, C11 - Noise, C23 Noise
    14, [1]; %Eyeblin
    15, [8]; %Eyeblin
    17, [4]; %Eyeblin
    };


% get subject directories.
dirs.home                   = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\';
dirs.log                    = fullfile(dirs.home,'Raw data','ICA');
cd(dirs.log);
filenameICs                 = dir(['ICA_','*_.mat']);
filenameICs                 = {filenameICs.name};
par.nSub                    = length(filenameICs);



%% loop over subjects.
for iSub = 1:par.nSub;
    
    filenamePostIC = fullfile(dirs.log,...
        [filenameICs{iSub}(5:end-5) 'postIC.mat']);
    
    if exist(filenamePostIC,'file')
        continue
    end
    
    % load data from previous step
    load(filenameICs{iSub})
    
    % remove ICs.
    cfg                     = [];
    cfg.component           = comp2remove{[comp2remove{:,1}]==par.sID,2};
    data                    = ft_rejectcomponent(cfg, comp);
        
   % save marks for trial rejection.
    save(filenamePostIC,'data','par')
    
end % end iSub-loop.

%% Go through trials a second time


dirs.home                   = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\';
dirs.log                    = fullfile(dirs.home,'Raw data','ICA');
cd(dirs.log);
filenamePostICs                 = dir(['*postIC.mat']);
filenamePostICs                 = {filenamePostICs.name};
par.nSub                    = length(filenamePostICs);


for iSub = 1:par.nSub
    
    rejects = [filenamePostICs{iSub}(1:11), '_rejected_trls_2']; %create filename for trl reject data
    load(filenamePostICs{iSub},'data','par');
    
    
    % skip subjects when preprocessing is already done.
    if exist(rejects,'file') ~= 2
        continue;
    end
    
    
    cfg = [];
%     cfg.preproc.lpfilter        = 'yes';
%     cfg.preproc.lpfreq          = 35
    cfg.viewmode                = 'vertical';
    cfg.allowoverlap            = 'yes';
    cfg.continuous              = 'no';
    cfg.plotlabels              = 'yes';
    cfg                         = ft_databrowser(cfg,data) 
    par.visualArtf              = cfg.artfctdef.visual.artifact;
    
    % THe authors of the original study did absolute rubbish here.
    % Apparently fieldtrip does not allow the usage of the
    % ft_rejectartifact function when there is overlap between the trials
    % (which is the case due to an excessively long baseline period).
    % Hence, they exported the fieldtrip data into the eeglab toolbox, used
    % the databrowser to mark artifactual trials manually. EEGlab then
    % outputs a binary array of the selcted trials, which then again was
    % re-imported into fieldtrip, and used in the ft_selectdata function to
    % reject the artifactual trials chosen in eeglab. Not sure if this is
    % the most elegant solution. I for my part will try to implement all of
    % it simply in fieldtrip, With a function that outputs me a binary
    % array. Saves a ton of time and lines of code I'd think.
    functions = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis'
    addpath(functions)
    rejected_trials2 = get_trlrej(par.visualArtf, data.sampleinfo);
    save(rejects,'rejected_trials2')
    
    % Note: participant 16 had extremely noisy data, over 160 trials had to
    % be removed and that was a conservative guess. Participant 15 was
    % similarly problematic. However, the problem was not excessive
    % movement, but drowsyness. The second part of the experiment was
    % cluttered with alpha oscillations. Removed ~70 trials, perhaps to
    % conservative as well.
end
%% Rejecting trials and interpolating channels (decided to interpolate)
% part 16 = [35, 50]; channels to interpolate

%% Import behavioral data - again, I use parts of the scripts used by the original authors

dirs.home                   = 'C:\Users\juliu\OneDrive\Dokumente\PhD-Thesis\Studying\Signal Processing Course\Experiment and Analysis\';
dirs.log                    = fullfile(dirs.home,'Raw data','ICA');
cd(dirs.log);
filenameICs                 = dir(['ICA_','*_.mat']);
filenameICs                 = {filenameICs.name};
par.nSub                    = length(filenameICs);


for iSub = 1:par.nSub;

% load data from previous step
    load(filenameFinalPP{iSub})
    
    % provide information on screen.
    sprintf('TF analysis of sID = %d',par.sID)
     %% Trialinformation.
    
    % retrieve trialindex of remaining EEG epochs.
    trlidx = find(~rejectedtrials);
    if ~isempty(rejectedtrials2)
        trlidx = trlidx(~rejectedtrials2);
    end
    if strcmpi(par.sID,'25'); trlidx = trlidx + 1; end %sID025 missed the first trigger - see EEGpav_1_preTR.m.
    if strcmpi(par.sID,'30'); trlidx = trlidx + 3; end %sID030 missed the first three triggers.
    
    % retrieve trialinfo from behavioural file.
    load(fullfile(par.dirs.log,'behaviour',sprintf('3017033.03_jesmaa_0%d_001',par.sID),...
        sprintf('3017033.03_jesmaa_0%d_001_results.mat',par.sID)))


    seq.stim        = [prep.seq.learn.stim{1}; prep.seq.learn.stim{2}];
    seq.resp        = [results.learn{1}.response; results.learn{2}.response];
    seq.RT          = [results.learn{1}.RT; results.learn{2}.RT];
    seq.outcome     = [results.learn{1}.outcome; results.learn{2}.outcome];
    seq.accuracy    = [results.learn{1}.acc; results.learn{2}.acc];
    seq.go          = [results.learn{1}.go; results.learn{2}.go];
    seq.splithalf   = [ones(160,1); 2*ones(160,1);ones(160,1); 2*ones(160,1)];
   
    % select trialinfo of EEG epochs.
    seq.EEG.trlidx  = trlidx;
    seq.EEG.stim    = seq.stim(trlidx);
    seq.EEG.resp    = seq.resp(trlidx);
    seq.EEG.RT      = seq.RT(trlidx);
    seq.EEG.outcome = seq.outcome(trlidx);
    seq.EEG.accuracy= seq.accuracy(trlidx);
    seq.EEG.go      = seq.go(trlidx);
    goCue                       = 1:4;
    winCue                      = [1 2 5 6];
    seq.EEG.action              =  ~ismember(seq.EEG.stim,goCue) + 1;
    seq.EEG.val                 = ~ismember(seq.EEG.stim,winCue) + 1;
    seq.EEG.fb                  = seq.EEG.outcome + 1;
    seq.EEG.fb(seq.EEG.val==2)  = seq.EEG.fb(seq.EEG.val==2)+1;
    seq.EEG.splithalf           = seq.splithalf(trlidx);
end
% IIIa - ERP Plotting

%% Preprocessing IV - Frequency Analysis