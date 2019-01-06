
%%
base = [pwd, '/'];
% addpath(genpath(base));

addpath(fullfile(base, 'utils'));

opt.mot = '/home/qi/benchmark';
% opt.mot = '/home/qi/projects';
opt.mot2d = 'MOT17Det';

% type = 'RFCN0.8';
% type = 'RFCN0';
% type = 'poi';
% type = 'yolo';
type = 'yolov3';
% type = 'RFCN';
% type = 'FRCNN';   % mot challenge

opt.result = '/home/qi/projects/LSTM_Qi_v27/py/detresults/';   % change path


opt.results =[opt.result,type];


seq_set_test = 'train';

% evaluation for all test sequences
benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);

% % FRCNN Detector
% seqs = {'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',...
%     'MOT17-10', 'MOT17-11', 'MOT17-13'};

detmap='c9-train.txt';
% detmap='c9-train_qi.txt';
qievaluateDetection(detmap, opt.results, benchmark_dir);
%%
%%
% % % % % test one 
% % % detmap='c9-train_qi.txt';
% % % evaluateDetection(detmap, opt.results, benchmark_dir);
% % 
% opt.mot = '/home/qi/benchmark';
% % opt.mot = '/home/qi/projects';
% opt.mot2d = 'MOT17Det';
% 
% opt.results = '/home/qi/projects/LSTM_Qi_v23/py/detresults';
% % opt.results = '/home/qi/projects/LSTM_Qi_v23/py/qidettest';
% 
% seq_set_test = 'train';
% 
% % evaluation for all test sequences
% benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);
% 
% detmap='c9-train_qi.txt';
% qievaluateDetection(detmap, opt.results, benchmark_dir);

