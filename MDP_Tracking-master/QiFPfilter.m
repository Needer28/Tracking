% testing MDP
% for current sequence.
is_show = 0;   % set is_show to 1 to show tracking results in testing 0
is_save = 1;   % set is_save to 1 to save tracking result
is_text = 1;   % set is_text to 1 to display detailed info 0
is_pause = 0;  % set is_pause to 1 to debug

% MOT17
numall = 7;
% MOT2015
% numall = 11;
% seq_idx = 1;  % 1 2 3 4 5 6 7                       % sequence
%%

seq_set = 'train';
% seq_set = 'test';

% load the trained model

% opt = globals();
opt = Qi_globals();

opt.is_text = is_text;
opt.exit_threshold = 0.7;


for seq_idx = 1:numall
    if strcmp(seq_set, 'train') == 1
        seq_name = opt.mot2d_train_seqs{seq_idx};
        seq_num = opt.mot2d_train_nums(seq_idx);
    else
        seq_name = opt.mot2d_test_seqs{seq_idx};
        seq_num = opt.mot2d_test_nums(seq_idx);
    end

    % build the dres structure for images
    filename = sprintf('%s/%s_dres_image.mat', opt.results, seq_name);
    if exist(filename, 'file') ~= 0
        object = load(filename);
        dres_image = object.dres_image;
        fprintf('load images from file %s done\n', filename);
    else
        dres_image = read_dres_image(opt, seq_set, seq_name, seq_num);
        fprintf('read images done\n');
%          save(filename, 'dres_image', '-v7.3');
    end

    % read detections
    filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'det', 'det.txt');
    dres_det = read_mot2dres(filename);

    if strcmp(seq_set, 'train') == 1
        % read ground truth
        filename = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'gt', 'gt.txt');
        dres_gt = read_mot2dres(filename);
        dres_gt = fix_groundtruth(seq_name, dres_gt);
    end


%     trackermodel = sprintf('results/%s_tracker.mat',seq_name);
    trackermodel = sprintf('results/%s%s_tracker.mat',opt.method,seq_name);
    
    object = load(trackermodel);                      % =============================
    tracker = object.tracker;

    detfilter = fullfile(opt.mot, opt.mot2d, seq_set, seq_name, 'det', 'detfilter.txt');

    fid = fopen(detfilter, 'wt');

    % intialize tracker
    I = dres_image.I{1};
    tracker = MDP_initialize_test(tracker, size(I,2), size(I,1), dres_det, is_show);

    % for each frame
    trackers = [];
    id = 0;


    for fr = 1:seq_num

        if is_text
            fprintf('frame %d\n', fr);
        else
            fprintf('.');
            if mod(fr, 100) == 0
                fprintf('\n');
            end        
        end

        % extract detection
        index = find(dres_det.fr == fr);
        dres = sub(dres_det, index);

        % nms
        boxes = [dres.x dres.y dres.x+dres.w dres.y+dres.h dres.r];
        index = nms_new(boxes, 0.6);
        dres = sub(dres, index);

        % nms
    %     boxes = [dres.x dres.y dres.x+dres.w dres.y+dres.h dres.r];
    %     index = nms_new(boxes, 0.6);
    %     dres = sub(dres, index);

        dres = MDP_crop_image_box(dres, dres_image.Igray{fr}, tracker);


        % find detections for initialization
        [dres, index] = generate_initial_index(trackers, dres);
        for i = 1:numel(index)
            % extract features
            dres_one = sub(dres, index(i));
            f = MDP_feature_active(tracker, dres_one);
            % prediction
            label = svmpredict(1, f, tracker.w_active, '-q');
            % make a decision
            if label < 0
                continue;
            end
            ind = index(i);
            % build the dres structure
            dres_one.fr = dres.fr(ind);
            dres_one.id = -1;
            dres_one.x = dres.x(ind);
            dres_one.y = dres.y(ind);
            dres_one.w = dres.w(ind);
            dres_one.h = dres.h(ind);
            dres_one.r = dres.r(ind);

            fprintf(fid, '%d,%d,%f,%f,%f,%f,%f\n', ...
                        dres_one.fr,dres_one.id,dres_one.x,dres_one.y,...
                            dres_one.w,dres_one.h,dres_one.r);

        end


    end

    fclose(fid);
end