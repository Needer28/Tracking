# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import numpy as np
from operator import itemgetter
from time import time
from bb_proc import get_iou, bb_update_vp2, ds_score, bb_update_vp, bb_pred, bb_pred_kcf,bb_pred_kcf_aff
from num2fname import num2fname
import cv2
import os

from deep_sort import nn_matching
from deep_sort.detection import Detection

# %% from shiyan4.py FP Filter add response as aff++ to be continued

# motuse.py ---yolov3.py
# generate_detections.py  ---reid detections
# drawnewdatatracking.py ---draw tracking results
# 这里还是全部处理的方式，如果直接把motuse.py和generate_detection.py导入，存在cuda_out_of memory的问题
# Here is the important code to test det&track using yolov3. 食堂视频

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
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
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        #if bbox[3] < min_height:
        #    continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list
# %%
# %%
if __name__ == "__main__":

    print('Here we train!\n')
    fpath = '/home/qi/benchmark/MOTQi/'

    threshold_l = 0.3  # low detection threshold
    threshold_h = 0.5  # high detection threshold
    threshold_s = 0.0359  # score threshold

    theshold_s2 = 0.3  # score threshold for id shorter than 7 frames
    n_init = 5  # time threshold

    time_cnt = 0
# %%
    # start tracking

    #foldername = ('dark', 'havingcar', 'moving','shitang')
    #length = (849, 900, 1574, 100)

    # folder = 'shitang'
    # res = (720, 1280)
    # l = 100

    folder ='dark'
    # res = (1920, 1080)
    res = (544, 960)
    l = 849

    print('Processing sequence: %s...' % folder)

    # detection data ================================================
    # os.system("python motuse.py")                         # use yolv3 to detection

    fname_det = '%s%s/det/det.txt' % (fpath, folder)

    dets = np.loadtxt(fname_det, delimiter=',')
    dets = dets.astype('float32')

    # os.system("python generate_detections.py")            # Re-ID feature extractor
    detection_file = '%s%s/det/%s.npy' % (fpath, folder, folder)

    sequence_dir = '%s%s' % (fpath, folder)
    seq_info = gather_sequence_info(sequence_dir, detection_file)

    min_confidence = 0.3
    nms_max_overlap = 1.0
    min_detection_height = 0
    max_cosine_distance = 0.2

    p_thr = 0.3 #0.25

    nn_budget = 100

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)

    start = time()

    id_active, id_inactive = [], []

    # for each frame
    for f_num in range(1, l + 1):
        curframe = f_num
        frame_idx = curframe
        print('current frame is %d\n' % curframe)

        # the detections of current frame =================================
        dets_f = dets[dets[:, 0] == f_num, :]
        dets_f = dets_f[dets_f[:, 6] > threshold_l, :]   # filter detection
        if dets_f.shape[0] == 0:
            continue

        # Load current frame generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence > threshold_l]
        #detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # indices = preprocessing.non_max_suppression(
        # boxes, nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]    # detections type: detection.Detection



        # add feature costmatrix ==========================================

        match_scores = np.zeros(dets_f.shape[0], dtype='float32')
        matched_flag = np.zeros(dets_f.shape[0], dtype=bool)

        # match_cosdistance = np.zeros(dets_f.shape[0], dtype='float32')   # zero initiation
        # match_eudistance = np.zeros(dets_f.shape[0], dtype='float32')

        match_cosdistance = np.ones(dets_f.shape[0], dtype='float32')     # one iinitiation
        # match_eudistance = np.ones(dets_f.shape[0], dtype='float32')

        id_updated  = []

        imgurl = '%s%s/img1/%s' % (fpath, folder, num2fname(f_num))
        image = cv2.imread(imgurl)

        for id_ in id_active:
            # if this id is too short to use lstm motion model
            if len(id_['bb']) < 7:

                # calculates the bb matching score
                # method1 iouscore max
                '''
                for det_num, det in enumerate(dets_f):
                    if matched_flag[det_num] == True:
                        match_scores[det_num] = 0
                    else:
                        match_scores[det_num] = get_iou(
                            id_['bb'][-1], det[2:6])[0][0]
                '''

                # method2 metric learning  min
                # using nn_matching
                for det_num, det in enumerate(dets_f):
                    if matched_flag[det_num] == True:
                        match_cosdistance[det_num] = 100   #inf
                        match_scores[det_num] = 0
                    else:
                        updateindex = det_num
                        detfeature = detections[updateindex].feature # (128,)

                        trackfeature = id_['features'][-1]          # (128,)

                        # Here to find the nearliest distance detection
                        dis_cos = nn_matching._nn_cosine_distance([trackfeature], [detfeature])
                        # dis_eu = nn_matching._nn_euclidean_distance([trackfeature], [detfeature])
                        match_cosdistance[det_num]=dis_cos

                # method1  max iou
                '''
                best_match = dets_f[match_scores.argmax()]
                best_match_score = match_scores.max()
                updateindex = match_scores.argmax()
                '''
                # method2  min distance
                minindex = match_cosdistance.argmin()
                best_match2 = dets_f[minindex]
                best_match_score2 = match_cosdistance.min()

                # matches the bb with highest score
                # matching successfully =========================================
                # method1
                '''
                if best_match_score >= threshold_s2:
                    bb_update_vp2(id_, best_match[2:6], res)
                    id_['bb'].append(best_match[2:6])
                    id_['max_score'] = max(id_['max_score'], best_match[-1])
                    id_['features'].append(detections[updateindex].feature)

                    matched_flag[match_scores.argmax()] = True
                    id_updated.append(id_)

                # finishes this id
                else:
                    # if it's a valid id
                    if (id_['max_score'] >= threshold_h and
                        len(id_['bb']) >= n_init):
                        id_inactive.append(id_)
                '''
                # method2
                if best_match_score2 < max_cosine_distance:
                    bb_update_vp2(id_, best_match2[2:6], res)
                    id_['bb'].append(best_match2[2:6])
                    id_['max_score'] = max(id_['max_score'], best_match2[-1])
                    id_['features'].append(detections[minindex].feature)

                    matched_flag[minindex] = True
                    id_updated.append(id_)

                # finishes this id
                else:
                    # if it's a valid id
                    if (id_['max_score'] >= threshold_h and
                                len(id_['bb']) >= n_init):
                        id_inactive.append(id_)
# // length(track)<7

            else:
                # LSTM
                # calculates the bb matching score
                for det_num, det in enumerate(dets_f):
                    if matched_flag[det_num] == True:
                        match_scores[det_num] = 0
                    else:
                        match_scores[det_num] = ds_score(
                            id_, det[2:6], res)[0][0]

                best_match = dets_f[match_scores.argmax()]
                best_match_score = match_scores.max()
                updateindex = match_scores.argmax()


                # matches the bb with highest score
                # matching successfully========================================
                if best_match_score >= threshold_s:

                    det = best_match[2:6]
                    peakvalue = bb_pred_kcf_aff(det, image)

                    detfeature = detections[updateindex].feature

                    trackfeature = id_['features'][-1]  # (128,)
                    dis_cos = nn_matching._nn_cosine_distance([trackfeature], [detfeature])

                    if peakvalue > p_thr:  # response  0.25
                        bb_update_vp(id_, best_match[2:6], res)
                        id_['bb'].append(best_match[2:6])
                        id_['max_score'] = max(id_['max_score'], best_match[-1])

                        id_['features'].append(detections[updateindex].feature)

                        matched_flag[match_scores.argmax()] = True
                        id_['pred'] = 0
                        id_updated.append(id_)

                # the id was not updated, predict the next bb, fix frame option
                elif id_['pred'] < 3:      # 6 4 missing detections
                    # print("KCFpredframe:%s" % f_num)

                    if f_num==749 and folder == 'MOT17-11-SDP':
                        print("give up frame %s" % f_num)
                        continue

                    peakvalue = bb_pred_kcf(id_, image)

                    # not updating max_score here
                    # bb_pred(id_, res) #original
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
                   'features':[detections[det_num].feature]}
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
# %%
    # now id_inactive is the final tracking result
    result_bb = []
    for id_num, id_ in enumerate(id_inactive):
        for bb_num, bb in enumerate(id_['bb']):
            result_bb += [[id_['f_start'] + bb_num, id_num + 1, bb[0], bb[1],
                           bb[2], bb[3], -1, -1, -1, -1]]
    result_bb.sort(key=itemgetter(1, 0))
    with open('/home/qi/benchmark/MOTQi/detandtrackingresults/%s.txt' % folder, 'w') as rst_f:
        for bb in result_bb:
            rst_f.write(','.join([str(value) for value in bb]) + '\n')

    print('Total tracking time consumption:', time_cnt, 's.')

    # os.system("python Qidrawtrackingbox.py")    # record tracking results
    # os.system("python Qidrawtrackingboxwithline.py")    # record tracking results
