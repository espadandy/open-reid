import argparse
import numpy as np
import torch
import os.path as osp


from PIL import Image
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature
from reid.utils.data import transforms as T

def euclidean_distance(features1, features2):
    diff = features1 - features2
    dist = torch.mm(diff, diff.t())
    dist = torch.sqrt(dist)
    return dist

def PIL2Tensor(image_PILs):
    """
        image_PILs: a list of PIL images

        output: a 4-D float tensor
    """
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.RectScale(256, 128),
        T.ToTensor(),
        normalizer,
    ])
    tensors = torch.cat(([transformer(image).unsqueeze(0) for image in image_PILs])) # PIL images -> a tensor batch
    return tensors

def extract_features(image_PILs, model):
    tsrs = PIL2Tensor(image_PILs)
    tsrs = tsrs.cuda()
    outputs = extract_cnn_feature(model, tsrs)
    return outputs

def extract_features_from_path(image_path, model=None):
    if model == None:
        model = models.create('resnet50', num_features=1024, dropout=0, num_classes=128)
        checkpoint = load_checkpoint('~/open_reid/logs/triplet_loss/cuhk03/resnet50/model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])

    # PIL image
    image_PIL = Image.open(image_path).convert('RGB')
    return extract_features([image_PIL], model)

def load_model(args):
    model = models.create(args.a, num_features=1024, dropout=args.dropout, num_classes=args.features)

    checkpoint = load_checkpoint(args.modellog + 'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    return model

def parse_arg():
    parser = argparse.ArgumentParser(description='CUHK3 model')
    parser.add_argument('--a', type=str, default='resnet50')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--img1', type=str)
    parser.add_argument('--img2', type=str)

    #working_dir = osp.dirname(osp.abspath(__file__))
    working_dir = '/home/yicong/open_reid/'
    #parser.add_argument('--track-dir', type=str, metavar='PATH',
    #                    default=osp.join(working_dir, 'examples/data/cuhk03/images/'))
    parser.add_argument('--track-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/video/video_image'))
    parser.add_argument('--modellog', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/triplet_loss/cuhk03/resnet50/'))

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arg()
    model = load_model(args)
    model = model.cuda()
    """
    maxval = 0
    minval = 9999999
    for i in range(10):
        for j in range(10):
            id1 = (8-len(args.person1)) * '0' + args.person1
            id2 = (8-len(args.person2)) * '0' + args.person2
            img1 = (4 - len(str(i))) * '0' + str(i)
            img2 = (4 - len(str(j))) * '0' + str(j)

            img1_path = (id1 + '_' + args.cam1 + '_' + img1 + '.jpg')
            img2_path = (id2 + '_' + args.cam2 + '_' + img2 + '.jpg')

            features1 = extract_features_from_path(osp.join(args.track_dir, path_img1), model)
            features2 = extract_features_from_path(osp.join(args.track_dir, path_img2), model)
            dist = euclidean_distance(features1, features2)
            print (dist[0][0])
            if args.img1 == args.img2 and dist[0][0] > maxval:
                maxval = dist[0][0]
            elif args.img1 != args.img2 and dist[0][0] < minval:
                minval = dist[0][0]
    if args.img1 == args.img2:
        print('same person, max distance: ', maxval)
    else:
        print('different person, min distance: ', minval)
    """

    features1 = extract_features_from_path(osp.join(args.track_dir, args.img1+'.jpg'), model)
    features2 = extract_features_from_path(osp.join(args.track_dir, args.img2+'.jpg'), model)
    dist = euclidean_distance(features1, features2)
    print (dist[0][0])


