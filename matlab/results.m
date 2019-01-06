fpath = '../MOT17/test/';
foldername = 'MOT17-14-SDP';
frame = 650;
img_name = [fpath foldername '/img1/' num2fname(frame)];
rst_name = ['./py/results/' foldername '.txt'];

rst = load(rst_name);
rst(:, 7:10) = [];
rst = rst(rst(:, 1) <= frame, :);
rst_current = rst(rst(:, 1) == frame, :);

color_table = [[1,0,0];[0,1,0];[0,0,1];[1,1,0];[1,0,1];[0,1,1];[1,1,1];...
               [0.745,0.5,0.063];[0.1255,0.58,1];...
               [0.9373,0.1882,0.90196];[0.2824,0.7333,0.75686];...
               [0.8196,0.15294,0.48627];[0.9608,0.4431,0.0118];...
               [0.6745,0.353,0.353];[0.502,0.502,0];[1,0.502,0.502]];
           
img = imread(img_name);
imshow(img);
hold on;

for bb_cnt = 1:size(rst_current, 1)
	bb = rst_current(bb_cnt, :);
    color_num = mod(bb_cnt,16);
    if color_num == 0
        color_num = 16;
    end
    % plot bb in the current frame
    x = [bb(3); bb(3); bb(3)+bb(5); bb(3)+bb(5); bb(3)];
    y = [bb(4); bb(4)+bb(6); bb(4)+bb(6); bb(4); bb(4)];
    plot(x, y, 'Color', color_table(color_num, :));
    % plot positions for the past frames
    rst_id = rst(rst(:, 2) == bb(2), :);
    x = rst_id(:, 3) + rst_id(:, 5) / 2;
    y = rst_id(:, 4) + rst_id(:, 6);
    plot(x, y, 'Color', color_table(color_num, :));
end

hold off;