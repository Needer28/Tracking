
% allseqs = {'MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
%     'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'};
% opt.mot = '../../';
% opt.mot = '/home/qi/projects';
% opt.mot2d = 'MOT17';
% opt.results = '/home/qi/projects/LSTM_Qi_v1/py/results';  % ORIGINAL
% opt.results = '/home/qi/projects/LSTM_Qi_v10/py/results';
% opt.results = '/home/qi/projects/LSTM_Qi_v20/py/results';
base = [pwd, '/'];
% addpath(genpath(base));

addpath(fullfile(base, 'utils'));

opt.mot = '/home/star/Desktop';
opt.mot2d = 'idea';
opt.results = '/home/star/Desktop/idea/LSTM_Qi_v27/py/results';


% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v1';   % original v1
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v2';   % original v1+kcf
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v3';   % original v1　remove add frame 6 all
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v4_02'; % kcf parm
seq_set_test = 'train';

% evaluation for all test sequences
benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);

% % FRCNN Detector
%seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',...
%     'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'};
seqs = {'MOT17-02-FRCNN', 'MOT17-09-FRCNN'};
%  
% evaluateTracking(seqs, opt.results, benchmark_dir);
% DPM BASELINE
% seqs = {'MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10',...
% 'MOT16-11', 'MOT16-13'};
evaluateTracking(seqs, opt.results, benchmark_dir);

% % DPM Detector
% seqs = {'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM',...
% 'MOT17-11-DPM', 'MOT17-13-DPM'};
% evaluateTracking(seqs, opt.results, benchmark_dir);


% SDP Detector
% seqs = {'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', ...
% 'MOT17-11-SDP', 'MOT17-13-SDP'};
% evaluateTracking(seqs, opt.results, benchmark_dir);

% % all detectors
% seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',...
% 'MOT17-11-FRCNN', 'MOT17-13-FRCNN', ...
%     'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP',...
% 'MOT17-13-SDP', ...
%     'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',...
% 'MOT17-13-DPM'};
% evaluateTracking(seqs, opt.results, benchmark_dir);



