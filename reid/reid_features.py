import cv2
import numpy as np
from distutils.version import LooseVersion
import torch
from torch.autograd import Variable

from reid import bbox as bbox_utils
from reid.log import logger
from reid import net_utils
from reid.image_part_aligned import Model


def load_reid_model():
    model = Model(n_parts=8)
    model.inp_size = (80, 160)
    ckpt = '../data/googlenet_part8_all_xavier_ckpt_56.h5'

    net_utils.load_net(ckpt, model)
    logger.info('Load ReID model from {}'.format(ckpt))
    # model = model.cuda() # model frome cpu->gpu
    model.eval()
    return model


def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = bbox_utils.clip_boxes(bboxes, image.shape)
    patches = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return patches


def extract_reid_features(reid_model, image, tlwhs):
    # t 表示目标边界框的左上角横坐标 l 表示目标边界框的左上角纵坐标
    # b 表示目标边界框的右下角横坐标 r 表示目标边界框的右下角纵坐标
    tlbrs = np.array([[tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]] for tlwh in tlwhs])
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs)
    patches = np.asarray([im_preprocess(cv2.resize(p, reid_model.inp_size)) for p in patches], dtype=np.float32)

    gpu = net_utils.get_device(reid_model)
    if LooseVersion(torch.__version__) > LooseVersion('0.3.1'):
        with torch.no_grad():
            im_var = Variable(torch.from_numpy(patches))
            if gpu is not None:
                im_var = im_var.cuda(gpu)
            features = reid_model(im_var).data
    else:
        im_var = Variable(torch.from_numpy(patches), volatile=True)
        if gpu is not None:
            im_var = im_var.cuda(gpu)
        features = reid_model(im_var).data

    return features
