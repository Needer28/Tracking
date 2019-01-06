
% allseqs = {'MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
%     'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'};
% opt.mot = '../../';

base = [pwd, '/'];
% addpath(genpath(base));

addpath(fullfile(base, 'utils'));

opt.mot = '/home/qi/projects';
opt.mot2d = 'MOT17';

% opt.results = '/home/qi/projects/LSTM_Qi_v1/py/results';  % ORIGINAL  60.6
% opt.results = '/home/qi/projects/LSTM_Qi_v10/py/results';   % 60.8 Yolo

opt.results = '/home/qi/projects/LSTM_Qi_v27/py/results';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/4.61.4';
% opt.results = '/home/qi/projects/iou/results';

% FN===================================================================
% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/paper/FP/no';
% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/paper/FP/yes';
% FN===================================================================
% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/4.61.4';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/8.60.0CNN';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/7.61.3_IOU+LSTM,removeCNN';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/6.60.2_KCFremove';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/SDP/9.61.2not pop';

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/POI';

%% CHANGE METHOD

% opt.results = '/home/qi/projects/LSTM_Qi_v27/shiyan/paper/best';








% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v1';   % original v1
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v2';   % original v1+kcf
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v3';   % original v1　remove add frame 6 all
% opt.results = '/home/qi/projects/LSTM_Qi_v5/py/results_v4_02'; % kcf parm
seq_set_test = 'train';

% evaluation for all test sequences
benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);

% %% POI Detector YOLOV3
% seqs = {'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',...
%     'MOT17-10', 'MOT17-11', 'MOT17-13'};
%  
% evaluateTracking(seqs, opt.results, benchmark_dir);
%%
% % % FRCNN Detector  % 46.8
% seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN',...
%     'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'};
% %  
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



