#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch
import torch.backends.cudnn as cudnn
from os.path import join
import time
import numpy as np
import random

from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.util_functions import Averaging, adjust_lr
from utils.util_functions import dir_check


def training(train_loader, epochs, save, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    logger.debug('create model')

    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    adjustable_lr = opt.lr

    logger.debug('epochs: %s', epochs)
    for epoch in range(epochs):
        model.cuda()
        model.train()

        logger.debug('Epoch # %d' % epoch)
        if opt.lr_adj:
            # if epoch in [int(epochs * 0.3), int(epochs * 0.7)]:
            # if epoch in [int(epochs * 0.5)]:
            if epoch % 30 == 0 and epoch > 0:
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
                logger.debug('lr: %f' % adjustable_lr)
        end = time.time()
        for i, (features, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            features = features.float().cuda(non_blocking=True)
            labels = labels.float().cuda()
            output = model(features)
            loss_values = loss(output, labels)
            losses.update(loss_values.item(), features.size(0))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        logger.debug('loss: %f' % losses.avg)
        losses.reset()

    if save:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(join(opt.dataset_root, 'models'))
        dir_check(join(opt.dataset_root, 'models', kwargs['name']))
        torch.save(save_dict, join(opt.dataset_root, 'models', kwargs['name'],
                                   '%s.pth.tar' % opt.log_str))
    return model


def load_model(name='mlp'):
    if opt.resume_str:
        subaction = opt.subaction.split('_')[0]
        resume_str = opt.resume_str % subaction
        # resume_str = opt.resume_str
    else:
        resume_str = opt.log_str
    checkpoint = torch.load(join(opt.dataset_root, 'models', name,
                                 '%s.pth.tar' % resume_str))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s.pth.tar' % resume_str)
    return checkpoint
