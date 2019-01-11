#!/usr/bin/env python

"""Creating dataset out of video features for different models.
"""

__all__ = ''
__author__ = 'Anna Kukleva'
__date__ = 'December 2018'


from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
import numpy as np
from os.path import join
import re

from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.util_functions import join_data


class FeatureDataset(Dataset):
    def __init__(self, features):
        logger.debug('Creating feature dataset')

        self._features = features
        self._gt = None
        self._videos = None

    def __len__(self):
        return len(self._gt)

    def __getitem__(self, idx):
        gt_item = self._gt[idx]
        features = self._features[idx]
        return np.asarray(features), gt_item


class GTDataset(FeatureDataset):
    def __init__(self, videos, features):
        logger.debug('Ground Truth labels')
        super().__init__(features)

        for video in videos:
            gt_item = np.asarray(video.gt).reshape((-1, 1))
            self._gt = join_data(self._gt, gt_item, np.vstack)


class RelTimeDataset(FeatureDataset):
    def __init__(self, videos, features):
        logger.debug('Relative time labels')
        super().__init__(features)

        temp_features = None  # used only if opt.concat > 1
        for video in videos:
            time_label = np.asarray(video.temp).reshape((-1, 1))
            video_features = self._features[video.global_range]
            if opt.rt_concat:
                video_features = join_data(video_features, time_label, np.hstack)

            if opt.concat > 1:
                video_features_concat = self._features[video.global_range].copy()
                last_frame = self._features[video.global_range][-1]

                for i in range(opt.concat, 1, -1):
                    video_features_concat = np.roll(video_features_concat, -1, axis=0)
                    video_features_concat[-1] = last_frame
                    video_features = join_data(video_features,
                                               video_features_concat,
                                               np.hstack)

            temp_features = join_data(temp_features, video_features, np.vstack)

            self._gt = join_data(self._gt, time_label, np.vstack)

        if opt.concat > 1 or opt.rt_concat:
            self._features = temp_features


class TCNDataset(FeatureDataset):
    def __init__(self, videos, features):
        logger.debug('Dataset for TCN model')
        super().__init__(features)
        self._videos = videos

        self._idx2videoidx = []
        self.l_in = opt.tcn_len

        for video_idx, video in enumerate(self._videos):
            gt_item = np.asarray(video.temp).reshape((-1, 1))
            self._gt = join_data(self._gt, gt_item, np.vstack)

            self._idx2videoidx += [video_idx] * video.n_frames

    def __getitem__(self, idx):
        tcn_features_out = np.zeros((self.l_in, opt.feature_dim))
        gt_out = np.zeros(self.l_in)

        video_idx = self._idx2videoidx[idx]
        video_start = self._videos[video_idx].global_start
        tcn_start = abs(min(0, idx - video_start - self.l_in))
        tcn_features_out[max(tcn_start, tcn_start-1):] = self._features[idx - self.l_in + tcn_start + 1: idx + 1]
        gt_out[max(tcn_start, tcn_start-1):] = self._gt[idx - self.l_in + tcn_start + 1: idx + 1].squeeze()
        if opt.rt_concat:
            tcn_features_out = join_data(tcn_features_out, gt_out.reshape((-1, 1)), np.hstack)
        return tcn_features_out.T, gt_out


class GlobalDataset(FeatureDataset):
    def __init__(self, assemblage, actions):
        logger.debug('Dataset for global model')
        super().__init__(None)

        # todo: check if assembledge passed by reference or not
        for action in actions:
            for video in assemblage[action].get_videos():
                rel_time = np.asarray(video.temp).reshape((-1, 1))
                self._gt = join_data(self._gt, rel_time, np.vstack)
            self._features = join_data(self._features,
                                       assemblage[action].get_features(),
                                       np.vstack)




def load_ground_truth(videos, features, shuffle=True):
    logger.debug('load data with ground truth labels for training some embedding')

    dataset = GTDataset(videos, features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)
    return dataloader


def load_reltime(videos=None, features=None, shuffle=True, **kwargs):
    logger.debug('load data with temporal labels as ground truth')
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.model_name == 'mlp':
        dataset = RelTimeDataset(videos, features)
    if opt.model_name == 'tcn':
        dataset = TCNDataset(videos, features)
    if opt.model_name == 'global':
        dataset = GlobalDataset(kwargs['assemblage'], kwargs['actions'])
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=shuffle,
                                             num_workers=opt.num_workers)

    return dataloader
