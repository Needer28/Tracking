clear; clc;

fpath = '../MOT17/train/';
foldername = ['MOT17-02-FRCNN'; 'MOT17-04-FRCNN'; 'MOT17-05-FRCNN';
    'MOT17-09-FRCNN'; 'MOT17-10-FRCNN'; 'MOT17-11-FRCNN'; 'MOT17-13-FRCNN'];

seqnum = size(foldername, 1);
all_bb = [];
for i = 1:seqnum
    gt_fname = [fpath foldername(i, :) '/gt/gt.txt'];
    dets = load(gt_fname);

    % uses valid pedestrian data only to train
    dets = dets(dets(:, 7)==1 & dets(:, 8)==1 & dets(:, 9)==1, :);
    dets = dets(dets(:, 5)<60 & dets(:, 6)<180 & dets(:, 5)>50 & dets(:, 6)>160, :);
    all_bb = [all_bb; dets(:, 5:6)];
end

hist3(all_bb, [9, 19]);
xlabel('Width (pixels)'); ylabel('Height (pixels)');
set(gcf,'renderer','opengl');
set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
view(2);

% the result is (52, 170)