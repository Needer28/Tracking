from PIL import Image, ImageDraw
import numpy as np
import os
from PIL import ImageFont
import cv2
from matplotlib import pyplot as plt

fpath = '/home/qi/benchmark/MOTQi/'

# foldername = ('dark', 'havingcar', 'moving', 'shitang')
# length = (849, 900, 1574,100)

folder = 'shitang'
res = (720, 1280)
l = 100

# res = (1920, 1080)

# folder = 'dark'
# res = (544, 960)
# l = 849

print('Draw sequence: %s...' % folder)

# detection data ================================================
# fname_track = '%s%s/%s.txt' % (fpath, folder,folder)

fname_track = '/home/qi/benchmark/MOTQi/detandtrackingresults/%s.txt' % folder

dets = np.loadtxt(fname_track, delimiter=',')
dets = dets.astype('float32')
k = res[0] - 100
boxes_traiectory = []
# for each frame
for f_num in range(1, l + 1):
    dets_f = dets[dets[:, 0] == f_num, :]
    if f_num < 10 and f_num > 0:
        pic = '00000' + str(f_num) + '.jpg'
    elif f_num < 100:
        pic = '0000' + str(f_num) + '.jpg'
    elif f_num < 1000:
        pic = '000' + str(f_num) + '.jpg'
    else:
        pic = '00' + str(f_num) + '.jpg'

    inputpic = '%s%s/img1/%s' % (fpath, folder, pic)
    outputpic = '%s%s/img2/%s' % (fpath, folder, pic)


    im = Image.open(inputpic)

    draw = ImageDraw.Draw(im)

    font = ImageFont.truetype('LiberationSans-Regular.ttf', 20)

    font1 = ImageFont.truetype('LiberationSans-Regular.ttf', 30)

    draw.text((k, 10), str(f_num), 'yellow', font=font1)



    for det_num, det in enumerate(dets_f):


        bb = det[1:6]
        id = bb[0]
        tid = int(id)
        ttid = str(tid)

        x = bb[1]
        y = bb[2]
        w = bb[3]
        h = bb[4]

        x1 = int(x + w / 2)
        x2 = int(y + h)
        center = (x1, x2)
        boxes_traiectory.append(center)

        for pos_tra in boxes_traiectory:
         #   draw.point([(int(pos_tra[0]), int(pos_tra[1]))], fill='red')
            draw.point([(int(pos_tra[0]), int(pos_tra[1]))], fill=(255, 255, 255))




        line = 6  # 3  change line bold

        if tid % 2 == 0:

            for i in range(1, line + 1):
                draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='red')

            # draw.rectangle((x, y, x + w, y + h), outline='red')
            draw.text((x, y), ttid, 'red', font=font)


        elif tid % 3 == 0:

            for i in range(1, line + 1):
                draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='green')

            # draw.rectangle((x, y, x + w, y + h), outline='green')
            draw.text((x, y), ttid, 'green', font=font)

        elif tid % 5 == 0:

            for i in range(1, line + 1):
                draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='blue')

            # draw.rectangle((x, y, x + w, y + h), outline='blue')
            draw.text((x, y), ttid, 'blue', font=font)

        elif tid % 7 == 0:

            for i in range(1, line + 1):
                draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='yellow')

            # draw.rectangle((x, y, x + w, y + h), outline='yellow')
            draw.text((x, y), ttid, 'yellow', font=font)


        else:

            for i in range(1, line + 1):
                draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='purple')

            # draw.rectangle((x, y, x + w, y + h), outline='purple')
            draw.text((x, y), ttid, 'purple', font=font)




    im.save(outputpic)


