% nms

ov_threshold = 0.4;

% fpath = '../MOT17/train/';
% foldername = ['MOT17-02-DPM'; 'MOT17-04-DPM'; 'MOT17-05-DPM';
%     'MOT17-09-DPM'; 'MOT17-10-DPM'; 'MOT17-11-DPM'; 'MOT17-13-DPM'];
fpath = '../MOT17/test/';
foldername = ['MOT17-01-DPM'; 'MOT17-03-DPM'; 'MOT17-06-DPM';
    'MOT17-07-DPM'; 'MOT17-08-DPM'; 'MOT17-12-DPM'; 'MOT17-14-DPM'];

seqnum = size(foldername, 1);
for a = 1:seqnum
    gt_fname = [fpath foldername(a, :) '/det/det.txt'];
    dets_all = load(gt_fname);
    dets_all(:, 8:10) = [];
    
    dets_save = [];
    
    f_max = max(dets_all(:,1));
    for f = 1:f_max
        dets = dets_all(dets_all(:,1) == f,:);
        if size(dets, 1) < 1
            continue;
        end
    
        obsNo = size(dets, 1);
        obsInd = 1:obsNo;
        overlap_mat = zeros(obsNo,obsNo);
        for j = 1:obsNo
            for k = 1:obsNo
                if j == k
                    continue;
                end
                overlap_mat(j, k) = iou(dets(j, 3:6), dets(k, 3:6));
            end
        end
        overlap_mat(overlap_mat < ov_threshold) = 0;
        [Matching, cost] = Hungarian(-overlap_mat);
        [row,col] = find((Matching==1) & (overlap_mat >= ov_threshold));
    
        % when overlapped, pick more confident detection
        obsIndDel = [];
        for i = 1:length(row)
            if dets(row(i), 7) > dets(col(i), 7)
                obsIndDel = [obsIndDel; col(i)];
            else
                obsIndDel = [obsIndDel; row(i)];
            end
        end

        dets(obsIndDel, :) = [];
        dets_save = [dets_save; dets];
        
    end
    
    fid = fopen(gt_fname, 'wt');
    [m, n] = size(dets_save);
    for i = 1:m
        for j = 1:n
            if j == n
                fprintf(fid, '%f\n', dets_save(i, j));
            else
                fprintf(fid, '%f,', dets_save(i, j));
            end
        end
    end
    fclose(fid);

end

