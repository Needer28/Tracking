function draw_bbs(fname, bbs, color_table)

img = imread(fname);
imshow(img);
hold on;

for bb_cnt = 1:size(bbs, 1)
	bb = bbs(bb_cnt, :);
    x = [bb(3), bb(3), bb(3)+bb(5), bb(3)+bb(5), bb(3)];
    y = [bb(4), bb(4)+bb(6), bb(4)+bb(6), bb(4), bb(4)];
    plot(x, y, 'Color', color_table(1+mod(bb(2),11), :));
end

hold off;