#!/usr/bin/env python

"""Inherited corpus, since we don't use ground truth labels to separate videos
into true action classes but into sets given by clusterization on the higher
level in hierarchy.
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2018'


import numpy as np
import torch

from corpus import Corpus
from utils.arg_pars import opt
from utils.logging_setup import logger


class LocalCorpus(Corpus):
    def __init__(self, videos, features, K=None, embedding=None):
        # todo: define bunch of parameters for the corpus
        subaction = ''
        logger.debug('%s' % subaction)

        super().__init__(Q=opt.gmm, subaction=subaction, K=K)

        self._videos = list(np.array(videos).copy())
        if embedding is not None:
            self._features = features.copy()
            self._embedding = embedding
        else:
            self._features = features[:, :opt.feature_dim]

        self._update_fg_mask()

    def _init_videos(self):
        logger.debug('nothing should happen')

    def pipeline(self, iterations=1, epochs=30, dim=20, lr=1e-3):
        opt.epochs = epochs
        opt.resume = False
        opt.embed_dim = dim
        opt.lr = lr
        if self._embedding is None:
            self.regression_training()
        else:
            self._embedded_feat = torch.Tensor(self._features)
            self._embedded_feat = self._embedding.embedded(self._embedded_feat).detach().numpy()

        self.clustering()

        for iteration in range(iterations):
            logger.debug('Iteration %d' % iteration)
            self.iter = iteration

            self.gaussian_model()
            self.accuracy_corpus()

            if opt.viterbi:
                self.ordering_sampler()
                self.rho_sampling()

                self.viterbi_decoding()
            else:
                self.subactivity_sampler()

                self.ordering_sampler()
                self.rho_sampling()

        self.accuracy_corpus()

    def pr_gt(self, pr_idx_start):
        long_pr = []
        long_gt = []

        for video in self._videos:
            long_pr += list(video._z)
            long_gt += list(video.gt)

        if opt.bg:
            long_pr = map(lambda x: x + pr_idx_start if x != -1 else x, long_pr)
        else:
            long_pr = np.asarray(long_pr) + pr_idx_start

        return list(long_pr), long_gt

    def stat(self):
        return self.return_stat




