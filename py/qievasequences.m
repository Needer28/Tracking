
% allseqs = {'MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
%     'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'};
% opt.mot = '../../';

base = [pwd, '/'];
% addpath(genpath(base));

addpath(fullfile(base, 'utils'));

opt.mot = '/home/qi/projects';
opt.mot2d = 'MOT17';

% opt.results = '/home/qi/projects/LSTM_Qi_v1/py/results';  % ORIGINAL  60.6   DPM 21.4
% opt.results = '/home/qi/projects/LSTM_Qi_v10/py/results';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/resultskalmanfilter';
opt.results = '/home/qi/projects/LSTM_Qi_v27/py/results';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/MOTDT17';       % sdp 57.7
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/HDTR_17';       % sdp 53.6
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/jCC';           % sdp 64.5
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/FWT_17';        % sdp 61.8
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/MHT_DAM_17';    % sdp 64.6 
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/EDMT17';        % sdp 64.1
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/IOU17';         % sdp 60.4
% opt.results = '/home/qi/projects/LSTM_Qi_v27/py/HAM_SADF17';    % sdp 63.4

% opt.results = '/home/qi/benchmark/qiresults/train';    %submit




% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/4.61.4';
% opt.results = '/home/qi/projects/iou/results';

% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v1';   % original v1
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v2';   % original v1+kcf
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v3';   % original v1　remove add frame 6 all
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v4_02'; % kcf parm
seq_set_test = 'train';

% evaluation for all test sequences
benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);

% %% POI Detector
% seqs = {'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',...
%     'MOT17-10', 'MOT17-11', 'MOT17-13'};
%  
% evaluateTracking(seqs, opt.results, benchmark_dir);
%%
% % FRCNN Detector  % 46.8
% seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',...
%     'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'};
% evaluateTracking(seqs, opt.results, benchmark_dir);

% % % DPM Detector
% seqs = {'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM',...
% 'MOT17-11-DPM', 'MOT17-13-DPM'};
% evaluateTracking(seqs, opt.results, benchmark_dir);


% % % SDP Detector
seqs = {'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', ...
'MOT17-11-SDP', 'MOT17-13-SDP'};
evaluateTracking(seqs, opt.results, benchmark_dir);

% % all detectors
% seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',...
% 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ...
%     'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP',...
% 'MOT17-13-SDP', ...
%     'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',...
% 'MOT17-13-DPM'};
% evaluateTracking(seqs, opt.results, benchmark_dir);
