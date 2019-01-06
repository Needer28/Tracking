function myonlytest
% seqs = {'TUD-Stadtmitte', 'TUD-Campus', 'PETS09-S2L1', ...
%    'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ADL-Rundle-6', ...
%    'ADL-Rundle-8', 'KITTI-13', 'KITTI-17', 'Venice-2'};
    opt = globals();
    mot2d_train_seqs = {'TUD-Stadtmitte', 'TUD-Campus', 'PETS09-S2L1', ...
       'ETH-Bahnhof', 'ETH-Sunnyday', 'ETH-Pedcross2', 'ADL-Rundle-6', ...
       'ADL-Rundle-8', 'KITTI-13', 'KITTI-17', 'Venice-2'};
%     seq_idx_train = {{1,2},{3}, {4,5,6}, {7,8}, {9,10},{11}};
%     seq_idx_test  = {{1}, {2}, {3}, {4}, {5}, {6},{7},{8},{9},{10},{11}};

    seq_idx_train = {{1}};
    seq_idx_test  = {{2}};
    seq_set_test = 'train';
    
    N = numel(seq_idx_train);
    idx_train = seq_idx_train{N};
    
    
    % load tracker from file
    filename = sprintf('%s/%s_tracker.mat', opt.results, mot2d_train_seqs{idx_train{end}});
    object = load(filename);
    tracker = object.tracker;
    fprintf('load tracker from file %s\n', filename);
    
    % testing
    testnum = numel(seq_idx_test);
    for i = 1:testnum
        idx_test = seq_idx_test{i};
        % number of testing sequences
        num = numel(idx_test);
        for j = 1:num            
            MDP_test(idx_test{j}, seq_set_test, tracker);
        end
    
    end
    