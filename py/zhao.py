# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import numpy as np
from operator import itemgetter
from time import time
from bb_proc import get_iou, bb_update_vp2, ds_score, bb_update_vp, bb_pred, bb_pred_kcf, bb_pred_kcf_aff
from num2fname import num2fname
import cv2
import os
import matplotlib.pyplot as plt
from deep_sort import nn_matching
from deep_sort.detection import Detection

from reid.reid_features import load_reid_model, extract_reid_features


# copy from shiyan4.py 去掉cf app
def gather_sequence_info(sequence_dir, detection_file):
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}

    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]

        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


if __name__ == "__main__":
    reid_model = load_reid_model()  # load reid model
    chosekind = 'FRCNN'
    if chosekind == 'FRCNN':
        print('current detection:%s\n' % chosekind)
        test = False
        if test:
            print('Here we test!\n')
            fpath = '/home/star/Desktop/idea/test/'
            foldername = ('MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN',
                          'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN',
                          'MOT17-14-FRCNN')
            resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
                          (1920, 1080), (1920, 1080), (1920, 1080))
            length = (450, 1500, 1194, 500, 625, 900, 750)
        else:
            print('Here we train!\n')
            fpath = '/home/star/Desktop/idea/train/'
            foldername = ('MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
                          'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN',
                          'MOT17-13-FRCNN')
            resolution = ((1920, 1080), (1920, 1080), (640, 480), (1920, 1080),
                          (1920, 1080), (1920, 1080), (1920, 1080))
            length = (600, 1050, 837, 525, 654, 900, 750)

        threshold_l = 0.6  # low detection threshold  原来是：0
        threshold_h = 0.9  # high detection threshold
        threshold_s = 0.0377  # score threshold
        threshold_s2 = 0.4  # score threshold for id shorter than 7 frames
        n_init = 4  # time threshold
        alpha = 1
        beta = 1
        theta = 0.2  # 0.1 0.08 47.4  0.2 47.5 0.3(pass) 0.25(pass 47.4)

    time_cnt = 0

    # start tracking
    for folder, res, l in zip(foldername, resolution, length):
        print('Processing sequence: %s...' % folder)

        # detection data ================================================
        fname_det = '%s%s/det/det.txt' % (fpath, folder)
        dets = np.loadtxt(fname_det, delimiter=',')  # 检测器检测到的所有帧结果
        dets = dets.astype('float32')
        detection_file = '%s%s/det/%s.npy' % (fpath, folder, folder)
        sequence_dir = '%s%s' % (fpath, folder)
        seq_info = gather_sequence_info(sequence_dir, detection_file)

        min_confidence = 0.3
        nms_max_overlap = 1.0
        min_detection_height = 0
        max_cosine_distance = 0.2
        p_thr = 0.3  # 0.25

        start = time()

        id_active, id_inactive = [], []
        # for each frame
        for f_num in range(1, l + 1):
            curframe = f_num
            frame_idx = curframe  # int
            print('current frame is %d' % curframe)

            # the detections of current frame =================================
            imgurl = '%s%s/img1/%s' % (fpath, folder, num2fname(f_num))
            image = cv2.imread(imgurl)
            dets_f = dets[dets[:, 0] == f_num, :]  # 取出当前帧的检测结果
            dets_f = dets_f[dets_f[:, 6] > threshold_l, :]  # filter detection#取出置信度大于阈值的检测结果
            if dets_f.shape[0] == 0:
                continue

            # Load current frame generate detections.
            detections = create_detections(
                seq_info["detections"], frame_idx, min_detection_height)
            detections = [d for d in detections if d.confidence > threshold_l]

            # Run non-maxima suppression.
            # t 表示目标边界框的左上角横坐标 l 表示目标边界框的左上角纵坐标
            # w 表示目标边界框的宽度 h 表示目标边界框的高度
            boxes = np.array([d.tlwh for d in detections])
            print(boxes)

            # add feature costmatrix ==========================================
            match_scores = np.zeros(dets_f.shape[0], dtype='float32')
            matched_flag = np.zeros(dets_f.shape[0], dtype=bool)
            match_cosdistance = np.ones(dets_f.shape[0], dtype='float32')  # one initiation

            detfeatures = extract_reid_features(reid_model, image, boxes)  # MOTDT reid features
            id_updated = []
            for id_ in id_active:
                # if this id is too short to use lstm motion model
                if len(id_['bb']) < 7:

                    for det_num, det in enumerate(dets_f):
                        if matched_flag[det_num] == True:
                            match_cosdistance[det_num] = 100  # inf
                            match_scores[det_num] = 0
                        else:
                            updateindex = det_num
                            # detfeature = detections[updateindex].feature  # (128,) # deepsort reid features

                            detfeature = detfeatures[updateindex].numpy()
                            trackfeature = id_['features'][-1]  # (128,)
                            # Here to find the nearliest distance detection
                            dis_cos = nn_matching._nn_cosine_distance([trackfeature], [detfeature])
                            match_cosdistance[det_num] = dis_cos
                    minindex = match_cosdistance.argmin()
                    best_match2 = dets_f[minindex]
                    best_match_score2 = match_cosdistance.min()
                    # method2
                    if best_match_score2 < max_cosine_distance:
                        bb_update_vp2(id_, best_match2[2:6], res)
                        id_['bb'].append(best_match2[2:6])
                        id_['max_score'] = max(id_['max_score'], best_match2[-1])
                        id_['features'].append(detfeatures[minindex].numpy())

                        matched_flag[minindex] = True
                        id_updated.append(id_)
                    # finishes this id
                    else:
                        # if it's a valid id
                        if (id_['max_score'] >= threshold_h and
                                len(id_['bb']) >= n_init):
                            id_inactive.append(id_)
                else:
                    # LSTM
                    # calculates the bb matching score
                    for det_num, det in enumerate(dets_f):
                        if matched_flag[det_num] == True:
                            match_cosdistance[det_num] = 100
                            match_scores[det_num] = 0
                        else:

                            #detfeature = detections[det_num].feature  # (128,) # deepsort reid features
                            detfeature = detfeatures[det_num].numpy()

                            trackfeature = id_['features'][-1]  # (128,)

                            # Here to find the nearliest distance detection
                            dis_cos = nn_matching._nn_cosine_distance([trackfeature], [detfeature])

                            similarityapp = 1 - dis_cos

                            similaritymotion = ds_score(
                                id_, det[2:6], res)[0][0]

                            similarity = alpha * similarityapp * beta * similaritymotion

                            match_scores[det_num] = similarity

                    best_match = dets_f[match_scores.argmax()]
                    best_match_score = match_scores.max()
                    updateindex = match_scores.argmax()

                    # matches the bb with highest score
                    # matching successfully========================================
                    if best_match_score >= theta:

                        bb_update_vp(id_, best_match[2:6], res)
                        id_['bb'].append(best_match[2:6])
                        id_['max_score'] = max(id_['max_score'], best_match[-1])

                        id_['features'].append(detfeatures[updateindex].numpy())

                        matched_flag[match_scores.argmax()] = True
                        id_['pred'] = 0
                        id_updated.append(id_)

                    # the id was not updated, predict the next bb, fix frame option
                    elif id_['pred'] < 3:  # 6 4 missing detections
                        if f_num == 749 and folder == 'MOT17-11-SDP':
                            print("give up frame %s" % f_num)
                            continue
                        peakvalue = bb_pred_kcf(id_, image)
                        id_updated.append(id_)

                    # finishes this id
                    else:
                        # if it has pred, clear all pred. not clearing v, p list
                        for i in range(id_['pred']):
                            id_['bb'].pop()
                        id_['pred'] = 0
                        # if it's a valid id
                        if (id_['max_score'] >= threshold_h and
                                len(id_['bb']) >= n_init):
                            id_inactive.append(id_)

            # creates new ids   ==============================add detection feature   # reid feature

            id_new = [{'bb': [det[2:6]],
                       'v_list': np.zeros((6, 2), dtype='float32'),
                       'max_score': det[6],
                       'f_start': f_num,
                       'pred': 0,
                       'features': [detfeatures[det_num].numpy()]}
                      for det_num, det in enumerate(dets_f)
                      if matched_flag[det_num] == False]

            id_active = id_updated + id_new

        # =======================================================================================－－－－－－－－－－－－－－－－
        # finishes the remained ids
        for id_ in id_active:
            # if it has pred, clear all pred. not clearing v, p list
            for i in range(id_['pred']):
                id_['bb'].pop()
            id_['pred'] = 0

            # if it's a valid id
            if id_['max_score'] >= threshold_h and len(id_['bb']) >= n_init:
                id_inactive.append(id_)

        end = time()
        time_cnt += end - start
        # now id_inactive is the final tracking result
        result_bb = []
        for id_num, id_ in enumerate(id_inactive):
            for bb_num, bb in enumerate(id_['bb']):
                result_bb += [[id_['f_start'] + bb_num, id_num + 1, bb[0], bb[1],
                               bb[2], bb[3], -1, -1, -1, -1]]
        result_bb.sort(key=itemgetter(1, 0))
        with open('/home/star/Desktop/idea/LSTM_Qi_v27/py/results/%s.txt' % folder, 'w') as rst_f:
            for bb in result_bb:
                rst_f.write(','.join([str(value) for value in bb]) + '\n')

    print('Total tracking time consumption:', time_cnt, 's.')
