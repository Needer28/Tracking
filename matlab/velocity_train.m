clear; clc;

fpath = '../../train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
    'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];
resolution = [[1920, 1080]; [1920, 1080]; [640, 480]; [1920, 1080];...
    [1920, 1080]; [1920, 1080]; [1920, 1080]];
fps = [30; 30; 14; 30; 30; 30; 25];
split_factor = 0.7;
scale_factor = 180; % keeps the calculated velocities not too small

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

    % center coordinates
    c = [dets(:, 1:2) dets(:, 3) + dets(:, 5) / 2 ...
        dets(:, 4) + dets(:, 6) / 2];

    % scales the coordinates into [0, 1]
    c(:, 3) = c(:, 3) * scale_factor / resolution(i, 1);
    c(:, 4) = c(:, 4) * scale_factor / resolution(i, 2);

    % forms the samples
    max_id = max(c(:, 2));
    idv_x = zeros(1, 6, 2);
    idv_y = zeros(1, 2);
    sample_cnt = 1;
    for j = 1:max_id % for each id
        if sum(c(:, 2)==j) < 8 % we need 6 velocities for X and 1 for Y
            continue;
        end
        idc = c(c(:, 2)==j, :);
        idv_x_tmp = [];
        for k = 1:6
            idv_x_tmp = [idv_x_tmp; idc(k + 1, 3:4) - idc(k, 3:4)];
        end
        idv_y_tmp = idc(8, 3:4) - idc(7, 3:4);
        idv_x(sample_cnt, :, :) = idv_x_tmp;
        idv_y(sample_cnt, :) = idv_y_tmp;
        sample_cnt = sample_cnt + 1;
        for k = 1:size(idc, 1) - 8
            idv_x_tmp = [idv_x_tmp(2:6, :); idv_y_tmp];
            idv_x(sample_cnt, :, :) = idv_x_tmp;
            idv_y_tmp = idc(k + 8, 3:4) - idc(k + 7, 3:4);
            idv_y(sample_cnt, :) = idv_y_tmp;
            sample_cnt = sample_cnt + 1;
        end
    end

    % saves the matrixes
    save([fpath foldername(i, :) '/v/v_x.mat'], 'idv_x');
    save([fpath foldername(i, :) '/v/v_y.mat'], 'idv_y');

    split = fix(size(idv_y, 1) * split_factor);
    idv_x_train = idv_x(1:split, :, :);
    idv_y_train = idv_y(1:split, :);
    idv_x_test = idv_x(split + 1:size(idv_y, 1), :, :);
    idv_y_test = idv_y(split + 1:size(idv_y, 1), :);

    save([fpath foldername(i, :) '/v/v_x_train.mat'], 'idv_x_train');
    save([fpath foldername(i, :) '/v/v_y_train.mat'], 'idv_y_train');
    save([fpath foldername(i, :) '/v/v_x_test.mat'], 'idv_x_test');
    save([fpath foldername(i, :) '/v/v_y_test.mat'], 'idv_y_test');
    
    v_x_train = [v_x_train; idv_x_train];
    v_y_train = [v_y_train; idv_y_train];
    v_x_test = [v_x_test; idv_x_test];
    v_y_test = [v_y_test; idv_y_test];
end

save([fpath 'v/v_x_train.mat'], 'v_x_train');
save([fpath 'v/v_y_train.mat'], 'v_y_train');
save([fpath 'v/v_x_test.mat'], 'v_x_test');
save([fpath 'v/v_y_test.mat'], 'v_y_test');