from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(names, split_id, data_dir, height, width, batch_size, num_instances,
             workers, combine_trainval):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    roots = [osp.join(data_dir, name) for name in names]
    dataset_list = [datasets.create(name, root, split_id=split_id) for name, root in zip(names, roots)]

    train_set = [(d.trainval if combine_trainval else d.train) for d in dataset_list]
    val_set = [d.val for d in dataset_list]
    num_classes = (sum([i.num_trainval_ids for i in dataset_list]) if combine_trainval
    else sum([i.num_train_ids for i in dataset_list]))

    tmp_sampler = []
    for t in train_set:
        tmp_sampler += t

    train_set = [Preprocessor(t, root=d.images_dir, transform=train_transformer) for t, d in zip(train_set, dataset_list)]
    val_set = [Preprocessor(v, root=d.images_dir, transform=test_transformer) for v, d in zip(val_set, dataset_list)]
    test_set = [Preprocessor(list(set(d.query) | set(d.gallery)),
                     root=d.images_dir, transform=test_transformer) for d in dataset_list]

    train_set = ConcatDataset(train_set)
    val_set = ConcatDataset(val_set)
    test_set = ConcatDataset(test_set)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(tmp_sampler, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset_list, num_classes, train_loader, val_loader, test_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)

    dataset_name_list = ["cuhk03", "market1501"]
    dataset_list, num_classes, train_loader, val_loader, test_loader = \
        get_data(dataset_name_list, args.split, args.data_dir, args.height,
                 args.width, args.batch_size, args.num_instances, args.workers,
                 args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        if args.mini_epoch:
            args.epochs += start_epoch
        best_top1 = checkpoint['best_top1']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        metric.train(model, train_loader)
        print("Validation:")
        tmp = []
        for i in dataset_list:
            tmp += i.val
        evaluator.evaluate(val_loader, tmp, tmp, metric)
        print("Test:")
        tmp1 = []
        tmp2 = []
        for i in dataset_list:
            tmp1 += i.query
            tmp2 += i.gallery
        evaluator.evaluate(test_loader, tmp1, tmp2, metric)
        return

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 or args.mini_epoch else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    tmp = []
    for d in dataset_list:
        tmp += d.val
    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer)
        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(val_loader, tmp, tmp)             # modified

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, args.arch, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, args.arch, 'model_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    metric.train(model, train_loader)
    q = []
    g = []
    for d in dataset_list:
        q += d.query
        g += d.gallery
    evaluator.evaluate(test_loader, q, g, metric)           # modified


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--mini-epoch', action='store_true')
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    #working_dir = osp.dirname(osp.abspath(__file__))
    working_dir = '/home/yicong/open_reid/'
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/all_datasets/'))

    main(parser.parse_args())
