import argparse
import numpy as np
import torch
import cv2
import os.path as osp

from PIL import Image
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature
from reidutil import extract_features


def extract_img_patch(image, bbox, patch_shape = (256, 128)):
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        ratio = bbox[2] / float(bbox[3])
        if ratio < target_aspect:
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width
        else:
            new_height = bbox[2] / target_aspect
            bbox[1] -= (new_height - bbox[3]) / 2
            bbox[3] = new_height

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])
    image = Image.fromarray(image)  # cv2's image -> PIL's image
    return image


def video_features(video_path, det_file, model):
    """
    use a model to extract features from a video whose bounding boxes indicated by .txt file.
        video_path: the absolute path (contains file name) of the video
        det_feilt: the absolute path (contains file name) of the .txt file indicating information of bounding boxes
        model: pytorch network model
    """

    det_in = np.genfromtxt(det_file, delimiter = ',', dtype = np.float32)
    areas = det_in[:, 4] * det_in[:, 5]
    idx = np.where((areas > 1000) & (areas < 18000))[0]
    det_in = det_in[idx, :]
    cap = cv2.VideoCapture(video_path)
    nframe = 1
    det_out = []
    frame_indices = det_in[:, 0].astype(np.int32)                      # all frames contain people
    frame_max = frame_indices.max()
    model = model.cuda()

    while nframe <= frame_max:                                      # traverse all frames
        print (nframe)
        if not nframe in frame_indices:                                # the current frame does not have people.
            nframe+= 1
            ret, frame = cap.read()
            continue
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx = np.where(frame_indices == nframe)[0]          # 'idx' indicates which bounding boxes do the current frame has.
        rows = det_in[idx, :]                               # 'rows' selects out those bounding boxes in the current frame.


        PIL_patch = [extract_img_patch(frame, bbox) for bbox in rows[:, 2:6].copy()]
        features = extract_features(PIL_patch, model).numpy()
        det_out += [np.r_[(row, feature)] for row, feature in zip(rows, features)]
        nframe += 1
    return det_out


def load_model(args):
    model = models.create(args.a, num_features=1024, dropout=args.dropout, num_classes=args.features)

    checkpoint = load_checkpoint(args.modellog + 'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    return model


"""
0: frame #
1: id #
2, 3: up-left x and y
4, 5: height, width
"""


def parse_arg():
    parser = argparse.ArgumentParser(description='CUHK3 model')
    parser.add_argument('--a', type=str, default='resnet50')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--features', type=int, default=128)

    #working_dir = osp.dirname(osp.abspath(__file__))
    working_dir = '/home/yicong/open_reid/'
    parser.add_argument('--modellog', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/triplet_loss/cuhk03/resnet50/'))
    parser.add_argument('--track-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/video/'))

    args = parser.parse_args()
    return args


def reid_main():
    args = parse_arg()
    model = load_model(args)
    video_path = osp.join(args.track_dir, 'surv_40.avi')
    det_file = osp.join(args.track_dir, 'tracking/tracking40.txt')
    det_out = video_features(video_path, det_file, model)


if __name__ == "__main__":
    reid_main()


