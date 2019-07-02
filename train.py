import argparse
import os
import logging
import time
import warnings
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv
from network import get_model
from gluoncv.utils import LRScheduler, LRSequential
from loss import EastLoss
from icdar_mx import get_batch
from config import Param
import ipdb


def parse_args():
    parser = argparse.ArgumentParser(
        description='train east with random shape')
    parser.add_argument('--train_config', required=True, type=str,
                        help='you need a yaml file for configure train')
    # parser.add_argument('--network', type=str, default='resnet50',
    #                     help='base network name which serves as feature extraction base.')
    # parser.add_argument('--trainset_path', type=str, default='', help='anno path for dataset')
    # parser.add_argument('--valset_path', type=str, default='', help='anno path for val dataset')
    # parser.add_argument('--data-shape', type=int, default=512, help='input data shape for evaluation')
    # parser.add_argument('--batch-size', type=int, default=32, help='training mini-batch size')
    # parser.add_argument('--num-workers', type=int, default=8, help='number of data workers')
    # parser.add_argument('--gpus', type=str, default='0', help='training with gpus, you can specify 1,3 for example')
    # parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    # parser.add_argument('--resume', type=str, default='', help='resume from previously saved parameters if not None')
    # parser.add_argument('--start-epoch', type=int, default=0, help='starting epoch for resuming')
    # parser.add_argument('--lr', type=float, default=0.0005, help="learn rate")
    # parser.add_argument('--lr-mode', type=str, default='step',
    #                     help='learning rate scheduler mode. option are step, polym cosine')
    # parser.add_argument('--lr-decay', type=float, default=0.1, help='decay rate of learning')
    # parser.add_argument('--lr-decay-period', type=int, default=0, help='interval for periodic learning rate decays')
    # parser.add_argument('--lr-decay_epoch', type=str, default='10,100,160', help='epochs at which learning rate decays.')
    # parser.add_argument('--warmup-lr', type=float, default=0.0, help='starting warmup learning rate.')
    # parser.add_argument('--warmup-epochs', type=float, default=0, help='number of warmup epoches')
    # parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
    # parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    # parser.add_argument('--log-interval', type=int, default=100, help='Logging mini-batch interval.')
    # parser.add_argument('--save-prefix', type=str, default='', help='saveing parameter prefix')
    #
    # parser.add_argument('--save-interval', type=int, default=10, help='Saving parameters epoch interval')
    # parser.add_argument('--val-interval', type=int, default=1, help='Epoch interval for validation')
    # parser.add_argument('--seed', type=int, default=233, help='Random seed to be fixed.')
    # parser.add_argument('--num-samples', type=int, default=-1,
    #                     help='Training images. Use -1 to automatically get the number.')
    # parser.add_argument('--no-random-shape', action='store_true',
    #                     help='Use fixed size(data-shape) throughout the training')
    # parser.add_argument('--no-wd', action='store_true',
    #                     help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    args = parser.parse_args()
    return args


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map >= best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(
            prefix, epoch, current_map))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(
            prefix, epoch, current_map))


def train(net, train_data, val_data, ctx, args):
    net.collect_params().reset_ctx(ctx)
    if args['lr_decay_period'] > 0:
        lr_decay_epoch = list(
            range(args['lr_decay_period'], args['epochs'], args['lr_decay_period']))
    else:
        lr_decay_epoch = [int(i) for i in args['lr_decay_epoch'].split(',')]
    lr_decay_epoch = [e - args['warmup_epochs'] for e in lr_decay_epoch]
    num_batches = args['num_sample'] // args['batch_size']
    lr_scheduler = LRSequential(
        [LRScheduler('linear', base_lr=0, target_lr=args['lr'], nepochs=args['warmup_epochs'], iters_per_epoch=num_batches),
         LRScheduler(args['lr_mode'], base_lr=args['lr'], nepochs=args['epoch'] - args['warmup_epochs'],
                     iters_per_epoch=num_batches,
                     step_epoch=lr_decay_epoch, step_factor=args['lr_decay'], power=2), ])

    #trainer = gluon.Trainer(net.collect_params(), 'sgd',
    #                        {'wd': args['wd'], 'momentum': args['momentum'], 'lr_scheduler': lr_scheduler, 'multi_precision': True},
    #                        kvstore='local')

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': args['lr'], 'multi_precision': True})
    east_loss = EastLoss()
    sum_losses = mx.metric.Loss('sum_loss')
    cls_losses = mx.metric.Loss('cls_loss')
    geo_losses = mx.metric.Loss('pos_loss')

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args['save_prefix'] + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args['start_epoch']))
    best_map = [0]

    for epoch in range(args['start_epoch'], args['epoch']):

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        # net.hybridize()

        for i in range(args['num_sample'] // args['batch_size']):
            batch = next(train_data)
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0)
            score_maps_gt = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0)
            geo_maps_gt = gluon.utils.split_and_load(
                batch[2], ctx_list=ctx, batch_axis=0)
            training_mask_gt = gluon.utils.split_and_load(
                batch[3], ctx_list=ctx, batch_axis=0)
            with autograd.record():
                for x in data:
                    cls_preds = []
                    geo_preds = []
                    score_maps_pred, geo_maps_pred = net(x)
                    cls_preds.append(score_maps_pred)
                    geo_preds.append(geo_maps_pred)
                sum_loss, cls_loss, geo_loss = east_loss(
                    score_maps_gt, cls_preds, geo_maps_gt, geo_preds, training_mask_gt)
            autograd.backward(sum_loss)
            trainer.step(batch_size)
            sum_losses.update(0, [l*batch_size for l in sum_loss])
            cls_losses.update(0, [l*batch_size for l in cls_loss])
            geo_losses.update(0, [l*batch_size for l in geo_loss])
            if (i + 1) % args['log_interval'] == 0:
                name1, loss1 = sum_losses.get()
                name2, loss2 = cls_losses.get()
                name3, loss3 = geo_losses.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, lr={:.5f}, {}={:.5f}, {}={:.5f}, {}={:.5f}'.format(epoch, i,
                                                                                                                                 batch_size *
                                                                                                                                 args['log_interval'] / (
                                                                                                                                     time.time() - btic),
                                                                                                                                 trainer.learning_rate,
                                                                                                                                 name1,
                                                                                                                                 loss1,
                                                                                                                                 name2,
                                                                                                                                 loss2,
                                                                                                                                 name3,
                                                                                                                                 loss3))
                btic = time.time()
        name1, loss1 = sum_losses.get()
        name2, loss2 = cls_losses.get()
        name3, loss3 = geo_losses.get()
        logger.info('[Epoch {}] Training cost: {:.5f}, {}={:.5f}, {}={:.5f}, {}={:.5f}'.format(epoch,
                                                                                               time.time() - tic,
                                                                                               name1,
                                                                                               loss1,
                                                                                               name2,
                                                                                               loss2,
                                                                                               name3,
                                                                                               loss3))

        net.save_parameters('{:s}_{:04d}.params'.format(
            args['save_prefix'], epoch))
        for i in range(args['num_val'] // args['batch_size']):
            batch = next(val_data)
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0)
            score_maps_gt = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0)
            geo_maps_gt = gluon.utils.split_and_load(
                batch[2], ctx_list=ctx, batch_axis=0)
            training_mask_gt = gluon.utils.split_and_load(
                batch[3], ctx_list=ctx, batch_axis=0)

            for x in data:
                cls_preds = []
                geo_preds = []
                score_maps_pred, geo_maps_pred = net(x)
                cls_preds.append(score_maps_pred)
                geo_preds.append(geo_maps_pred)

            sum_loss, cls_loss, geo_loss = east_loss(
                score_maps_gt, cls_preds, geo_maps_gt, geo_preds, training_mask_gt)
            sum_losses.update(0, [l*batch_size for l in sum_loss])
            cls_losses.update(0, [l*batch_size for l in cls_loss])
            geo_losses.update(0, [l*batch_size for l in geo_loss])
        name1, loss1 = sum_losses.get()
        name2, loss2 = cls_losses.get()
        name3, loss3 = geo_losses.get()
        logger.info('[Epoch {}],val loss: {}={:.5f}, {}={:.5f}, {}={:.5f}'.format(epoch,
                                                                                  name1,
                                                                                  loss1,
                                                                                  name2,
                                                                                  loss2,
                                                                                  name3,
                                                                                  loss3))


if __name__ == '__main__':
    system_arg = parse_args()
    args = Param()
    args.load_parm(system_arg.train_config)
    args = args.get_parm()

    gluoncv.utils.random.seed(args['seed'])
    ctx = [mx.gpu(int(i)) for i in args['gpus'].split(',') if i.strip()]
    print(ctx)
    ctx = ctx if ctx else [mx.cpu()]
    net_name = '_'.join(('east', str(args['data_shape']), args['network']))
    args['save_prefix'] += net_name
    net = get_model(args['network'], text_scale=args['data_shape'])
    async_net = net
    if args['resume']:
        net.load_parameters(args['resume'])
        async_net.load_parameters(args['resume'].strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
    #for item in net.features.collect_params().items():
    #    if item[0].split('_')[-1] not in ['gamma', 'beta', 'mean', 'var']:
    #        item[1].cast('float16')

    #for item in net.features.collect_params().items():
    #    if 'mobilenet' in item[0].split('_')[1]:
    #       item[1].lr_mult = 0.1
    train_loader = get_batch(
        num_workers=args['train_workers'], batch_size=args['batch_size'], data_flag='train', param=args)
    val_loader = get_batch(
        num_workers=args['eval_workers'], batch_size=args['batch_size'], data_flag='val', param=args)
    train(net, train_loader, val_loader, ctx, args)
