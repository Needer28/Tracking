from keras.models import load_model
from scipy import io
import numpy as np
from bb_proc import get_iou, get_v
import random


# ===================================================================== ONE
fpath = '../../../MOT17/train/'
print(fpath)
# len 7 //6 data a group
foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
              'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
              'MOT17-13-FRCNN')

# len 8 // 7 data a group
# foldername = ('MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP',
#               'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP',
#               'MOT17-13-SDP')

# # len 9 // 8 data a group
# foldername = ('MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM',
#               'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',
#               'MOT17-13-DPM')

resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
              (1920, 1080), (1920, 1080), (1920, 1080))
# foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN')
# resolution = ((1920, 1080), (1920, 1080))

ds_x = np.zeros((587599, 1), dtype='float32')
ds_y = np.zeros((587599, 1), dtype='float32')
ds_cnt = 0

v_model = load_model('/home/xie/Desktop/qi/MOT17/v_model.h5')
# p_model = load_model('p_model.h5')

for folder, res in zip(foldername, resolution):
    print(folder)
    fname = '%s%s/gt/gt2.mat' % (fpath, folder)
    bb_all = io.loadmat(fname)['dets']
    id_num = bb_all[:, 1].max()
    for id_cnt in range(1, id_num + 1):
        bb_id = bb_all[bb_all[:, 1] == id_cnt, :]
        if bb_id.shape[0] == 0:
            continue
        max_frame = bb_id[:, 0].max()
        min_frame=bb_id[:, 0].min()
        bb_list = np.zeros((7, 4))
        v_list = np.zeros((6, 2), dtype='float32')
        # p_list = np.zeros((6, 1), dtype='float32')
        v_list_4_pred = np.zeros((1, 6, 2), dtype='float32')
        # p_list_4_pred = np.zeros((1, 6, 1), dtype='float32')
        current_bb = np.zeros((1, 4))
        current_v = np.zeros((1, 2), dtype='float32')
        # current_p = np.zeros((1, 1), dtype='float32')
        neg_v = np.zeros((1, 2), dtype='float32')
        # neg_p = np.zeros((1, 1), dtype='float32')

        for f_cnt in range(min_frame, max_frame + 1):
            n = f_cnt - min_frame
            if n < 7:
                bb_list[n] = bb_id[n, 2:6]
                if n > 0:
                    v_list[n - 1] = get_v(bb_list[n - 1], bb_list[n], res)
                    # p_list[n - 1] = get_iou(bb_list[n - 1], bb_list[n])
            else:
                # calculates loss
                current_bb[0] = bb_id[n, 2:6]
                current_v[0] = get_v(bb_list[6], current_bb[0], res)
                # current_p[0] = get_iou(bb_list[6], current_bb[0])
                v_list_4_pred[0] = v_list
                # p_list_4_pred[0] = p_list

                v_loss = v_model.evaluate(
                    x=v_list_4_pred, y=current_v, batch_size=1, verbose=0)
                # p_loss = p_model.evaluate(
                    # x=p_list_4_pred, y=current_p, batch_size=1, verbose=0)
                v_loss = v_loss[0]

                # saves positive samples
                ds_x[ds_cnt] = np.array([v_loss], dtype='float32')
                ds_y[ds_cnt] = np.array([1], dtype='float32')
                ds_cnt += 1

                # saves negative samples
                bb_f = bb_all[bb_all[:, 0] == f_cnt, :]
                for m in range(bb_f.shape[0]):
                    if bb_f[m, 1] == id_cnt:
                        continue
                    # randomly discards most of negative samples
                    if random.randint(0, 99) > 6:
                        continue
                    neg_v[0] = get_v(bb_list[6], bb_f[m, 2:6], res)
                    # neg_p[0] = get_iou(bb_list[6], bb_f[m, 2:6])
                    v_loss = v_model.evaluate(
                        x=v_list_4_pred, y=neg_v, batch_size=1, verbose=0)
                    # p_loss = p_model.evaluate(
                        # x=p_list_4_pred, y=neg_p, batch_size=1, verbose=0)
                    v_loss = v_loss[0]
                    ds_x[ds_cnt] = np.array([v_loss], dtype='float32')
                    ds_y[ds_cnt] = np.array([0], dtype='float32')
                    ds_cnt += 1

                # saves current bb, v, p
                bb_list = np.delete(bb_list, (0), axis=0)
                v_list = np.delete(v_list, (0), axis=0)
                # p_list = np.delete(p_list, (0), axis=0)
                bb_list = np.append(bb_list, current_bb, axis=0)
                v_list = np.append(v_list, current_v, axis=0)
                # p_list = np.append(p_list, current_p, axis=0)

np.save('%sds_x.npy' % fpath, ds_x)
np.save('%sds_y.npy' % fpath, ds_y)

print("ds_tain is OK!")