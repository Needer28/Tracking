clear; clc;

fpath = '../MOT17/train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
    'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];
resolution = [[1920, 1080]; [1920, 1080]; [640, 480]; [1920, 1080];...
    [1920, 1080]; [1920, 1080]; [1920, 1080]];
fps = [30; 30; 14; 30; 30; 30; 25];
split_factor = 0.7;

p_x_train = [];
p_y_train = [];
p_x_test = [];
p_y_test = [];

seqnum = size(foldername, 1);
for i = 1:seqnum
    gt_fname = [fpath foldername(i, :) '/gt/gt.txt'];
    dets = load(gt_fname);

    % uses valid pedestrian data only to train
    dets = dets(dets(:, 7)==1 & dets(:, 8)==1, :);

    % keeps the coordinates info only
    dets = dets(:, 1:6);

    % forms the samples
    max_id = max(dets(:, 2));
    idp_x = zeros(1, 6, 1);
    idp_y = zeros(1, 1);
    sample_cnt = 1;
    for j = 1:max_id % for each id
        if sum(dets(:, 2)==j) < 8 % we need 6 velocities for X and 1 for Y
            continue;
        end
        idc = dets(dets(:, 2)==j, :);
        idp_x_tmp = [];
        for k = 1:6
            idp_x_tmp = [idp_x_tmp; iou(idc(k, 3:6), idc(k + 1, 3:6))];
        end
        idp_y_tmp = iou(idc(7, 3:6), idc(8, 3:6));
        idp_x(sample_cnt, :, :) = idp_x_tmp;
        idp_y(sample_cnt, :) = idp_y_tmp;
        sample_cnt = sample_cnt + 1;
        for k = 1:size(idc, 1) - 8
            idp_x_tmp = [idp_x_tmp(2:6, :); idp_y_tmp];
            idp_x(sample_cnt, :, :) = idp_x_tmp;
            idp_y_tmp = iou(idc(k + 7, 3:6), idc(k + 8, 3:6));
            idp_y(sample_cnt, :) = idp_y_tmp;
            sample_cnt = sample_cnt + 1;
        end
    end

    % saves the matrixes
    save([fpath foldername(i, :) '/p/p_x.mat'], 'idp_x');
    save([fpath foldername(i, :) '/p/p_y.mat'], 'idp_y');

    split = fix(size(idp_y, 1) * split_factor);
    idp_x_train = idp_x(1:split, :, :);
    idp_y_train = idp_y(1:split, :);
    idp_x_test = idp_x(split + 1:size(idp_y, 1), :, :);
    idp_y_test = idp_y(split + 1:size(idp_y, 1), :);

    save([fpath foldername(i, :) '/p/p_x_train.mat'], 'idp_x_train');
    save([fpath foldername(i, :) '/p/p_y_train.mat'], 'idp_y_train');
    save([fpath foldername(i, :) '/p/p_x_test.mat'], 'idp_x_test');
    save([fpath foldername(i, :) '/p/p_y_test.mat'], 'idp_y_test');
    
    p_x_train = [p_x_train; idp_x_train];
    p_y_train = [p_y_train; idp_y_train];
    p_x_test = [p_x_test; idp_x_test];
    p_y_test = [p_y_test; idp_y_test];
end

save([fpath 'p/p_x_train.mat'], 'p_x_train');
save([fpath 'p/p_y_train.mat'], 'p_y_train');
save([fpath 'p/p_x_test.mat'], 'p_x_test');
save([fpath 'p/p_y_test.mat'], 'p_y_test');