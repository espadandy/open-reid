from reid.utils.osutils import mkdir_if_missing

import argparse
import os.path as osp
import numpy as np
import os.path as osp
import cv2
from PIL import Image


def extract_img_patch(image, bbox, patch_shape = (256, 128)):
    """
        image: the frame
        bbox: a single bounding box in the frame
    """
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


def extract_imgs(video_path, det_file, args):
    det_in = np.genfromtxt(det_file, delimiter = ',', dtype = np.float32)
    areas = det_in[:, 4] * det_in[:, 5]
    idx = np.where((areas > 1000) & (areas < 18000))[0]
    det_in = det_in[idx, :]
    cap = cv2.VideoCapture(video_path)
    nframe = 1
    det_out = []
    frame_indices = det_in[:, 0].astype(np.int32)                      # all frames contain people
    frame_max = frame_indices.max()

    imgs_path = osp.join(args.video_dir, 'video_image')
    mkdir_if_missing(imgs_path)

    dic = {}
    while nframe <= frame_max:                                      # traverse all frames
        #print (nframe)
        if not nframe in frame_indices:                                # the current frame does not have people.
            nframe+= 1
            ret, frame = cap.read()
            continue
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx = np.where(frame_indices == nframe)[0]          # 'idx' indicates which bounding boxes do the current frame has.
        rows = det_in[idx, :]                               # 'rows' selects out those bounding boxes in the current frame.
        for bbox in rows[:, 1:6].copy():
            img = extract_img_patch(frame, bbox[1:])
            if str(int(bbox[0])) in dic:
                dic[str(int(bbox[0]))] += 1
            else:
                dic[str(int(bbox[0]))] = 0
            fname = '{:08d}_{:02d}_{:04d}.jpg'.format(int(bbox[0]), 0, dic[str(int(bbox[0]))])
            img.save(osp.join(imgs_path, fname), 'JPEG')
        nframe+= 1
    return True


def parse_arg():
    parser = argparse.ArgumentParser(description='Extract image from video.')
    working_dir = '/home/yicong/open_reid/'
    parser.add_argument('--video-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/video/'))
    args = parser.parse_args()
    return args


def extract_main():
    args = parse_arg()
    video_path = osp.join(args.video_dir, 'surv_40.avi')
    det_file = osp.join(args.video_dir, 'tracking/tracking40.txt')
    det_out = extract_imgs(video_path, det_file, args)


if __name__ == "__main__":
    extract_main()