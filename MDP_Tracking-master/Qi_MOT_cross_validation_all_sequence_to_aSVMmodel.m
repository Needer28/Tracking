% --------------------------------------------------------
% MDP Tracking
% Copyright (c) 2015 CVGL Stanford
% Licensed under The MIT License [see LICENSE for details]
% Written by Yu Xiang
% --------------------------------------------------------
%
% cross_validation on the MOT benchmark
function Qi_MOT_cross_validation_all_sequence_to_aSVMmodel

% set is_train to 0 if testing trained trackers only
% is_train = 0;

is_train = 1;
opt = Qi_globals();

method = opt.method;
switch method
    case 'DPM'
        mot2d_train_seqs = {'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', ...
            'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM', 'MOT17-13-DPM'};
         
    case 'FRCNN',
        mot2d_train_seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', ...
            'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'};
         
    case 'SDP',
        mot2d_train_seqs = {'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', ...
            'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP'};
              
    otherwise
        error('Unknown method.')
end

% mot2d_train_seqs = {'TUD-Stadtmitte', 'TUD-Campus', 'PETS09-S2L1', ...
%    'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ADL-Rundle-6', ...
%    'ADL-Rundle-8', 'KITTI-13', 'KITTI-17', 'Venice-2'};

% training and testing pairs
% seq_idx_train = {{2}, {4}, {1,3}};
% seq_idx_test  = {{5}, {6}, {7}};

% seq_idx_train = {{1,2,3,4,5,6,7}};
% seq_idx_test  = {{4}};

seq_idx_train = {{5},{2},{3},{4},{5},{6},{7}};
seq_idx_test  = {{1},{2},{3},{4},{5},{6},{7}};

seq_set_test = 'train';
N = numel(seq_idx_train);

% for each training-testing pair
for i = 1:N
    % training
    idx_train = seq_idx_train{i};
    
    if is_train
        % number of training sequences
        num = numel(idx_train);
        tracker = [];
        
        % online training
        for j = 1:num
            fprintf('Online training on sequence: %s\n', mot2d_train_seqs{idx_train{j}});
            tracker = MDP_train(idx_train{j}, tracker);
        end
        fprintf('%d training examples after online training\n', size(tracker.f_occluded, 1));
        
    else
        % set is_train to 0 if testing trained trackers only
        % load tracker from file
%         filename = sprintf('%s/%s_tracker.mat', opt.results, mot2d_train_seqs{idx_train{end}});
        filename = sprintf('%s/%s_tracker.mat', opt.results, opt.method);
        
        object = load(filename);
        tracker = object.tracker;
        fprintf('load tracker from file %s\n', filename);
    end
    
    % testing
    idx_test = seq_idx_test{i};
    % number of testing sequences
    num = numel(idx_test);
    for j = 1:num
        fprintf('Testing on sequence: %s\n', mot2d_train_seqs{idx_test{j}});
        MDP_test(idx_test{j}, seq_set_test, tracker);   % original MOTA=25.2
% MDP_test_hungarian(idx_test{j}, seq_set_test, tracker);   %change MOTA=25.0
    end    
end


benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);

switch method
    case 'DPM'
        seqs = {'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', ...
            'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM', 'MOT17-13-DPM'};
         
    case 'FRCNN',
        seqs = {'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', ...
            'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'};

    case 'SDP',
        seqs = {'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', ...
            'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP'};
              
    otherwise
        error('Unknown method.')
end


% seqs = {'MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', ...
%             'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM', 'MOT17-13-DPM'};
        
% seqs = {'MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', ...
%             'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP', 'MOT17-13-SDP'};
evaluateTracking(seqs, opt.results, benchmark_dir);

% evaluation for all test sequences
% benchmark_dir = fullfile(opt.mot, opt.mot2d, seq_set_test, filesep);
% seqs = {'TUD-Campus', 'ETH-Sunnyday', 'ETH-Pedcross2', ...
%    'ADL-Rundle-8', 'Venice-2', 'KITTI-17'};
% evaluateTracking(seqs, opt.results, benchmark_dir);