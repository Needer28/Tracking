function fname = num2fname(num)
% Turns an int num into MOT image file name.
% Example: input 32, output '000032.jpg'
    if num < 10 && num > 0
        fname = ['00000' num2str(num) '.jpg'];
    elseif num < 100
        fname = ['0000' num2str(num) '.jpg'];
    elseif num < 1000
        fname = ['000' num2str(num) '.jpg'];
    else
        fname = ['00' num2str(num) '.jpg'];
    end
end