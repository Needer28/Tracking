function draw_bb(fname, left, top, width, height)

img = imread(fname);
imshow(img);
hold on;
x = [left, left, left+width, left+width, left];
y = [top, top+height, top+height, top, top];
plot(x, y, 'r-');
hold off;