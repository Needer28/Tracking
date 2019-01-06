clear; clc;

fpath = '../../train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
    'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];

v_x_train = [];
v_y_train = [];
v_x_test = [];
v_y_test = [];

seqnum = size(foldername, 1);
for i = 1:seqnum
    gt_fname = [fpath foldername(i, :) '/gt/gt.txt'];
    dets = load(gt_fname);

    % uses valid pedestrian data only to train
    dets = dets(dets(:, 7)==1 & dets(:, 8)==1, :);
    
    dets(:, 7:9) = [];

    % forms the samples
    max_id = max(dets(:, 2));
    for j = 1:max_id % for each id
        if sum(dets(:, 2)==j) < 8 % we need 6 velocities for X and 1 for Y
            dets(dets(:, 2)==j, :) = [];
        end
    end
    
    dets = sortrows(dets, 2);

    % saves the matrixes
    save([fpath foldername(i, :) '/gt/gt2.mat'], 'dets');
end
figure