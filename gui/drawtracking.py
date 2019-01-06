from PIL import Image, ImageDraw
import numpy as np
import os
from PIL import ImageFont


fpath = '/home/qi/benchmark/MOT17/tracking/'
foldername = ('MOT17-02', 'MOT17-04', 'MOT17-05',
           'MOT17-09', 'MOT17-10', 'MOT17-11',
          'MOT17-13')
resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
            (1920, 1080), (1920, 1080), (1920, 1080))
length = (600, 1050, 837, 525, 654, 900, 750)

for folder, res, l in zip(foldername, resolution, length):
    print('Processing sequence: %s...' % folder)

    # detection data ================================================
    fname_track = '%s%s/%s.txt' % (fpath, folder,folder)
    dets = np.loadtxt(fname_track, delimiter=',')
    dets = dets.astype('float32')
    k = res[0]-100

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

            line = 6      #3

            if tid%2 == 0:

                for i in range(1, line + 1):
                    draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='red')
                # draw.rectangle((x, y, x + w, y + h), outline='red')
                draw.text((x, y), ttid, 'red',font=font)



            elif tid%3 == 0:

                for i in range(1, line + 1):
                    draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='green')
                # draw.rectangle((x, y, x + w, y + h), outline='green')
                draw.text((x, y), ttid, 'green',font=font)

            elif tid%5 == 0:

                for i in range(1, line + 1):
                    draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='blue')
                # draw.rectangle((x, y, x + w, y + h), outline='blue')
                draw.text((x, y), ttid, 'blue',font=font)

            elif tid%7 == 0:

                for i in range(1, line + 1):
                    draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='yellow')
                # draw.rectangle((x, y, x + w, y + h), outline='yellow')
                draw.text((x, y), ttid, 'yellow',font=font)
            else:

                for i in range(1, line + 1):
                    draw.rectangle((x + (line - i), y + (line - i), x + w + i, y + h + i), outline='purple')
                # draw.rectangle((x, y, x + w, y + h), outline='purple')
                draw.text((x, y), ttid, 'purple', font=font)




        im.save(outputpic)


'''
            inputpic = '/home/qi/Videos/1.jpg'
            outputpic='/home/qi/Videos/2.jpg'
            im = Image.open(inputpic)
            draw = ImageDraw.Draw(im)

            line = 5
            x, y = 10, 10
            width, height = 100, 50
            for i in range(1, line + 1):
                draw.rectangle((x + (line - i), y + (line - i), x + width + i, y + height + i), outline='red')

            im.save(outputpic)
'''
