#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'October 2018'


import numpy as np


class ActionVideo:
    def __init__(self, name, n_clust):
        self.n_clust = n_clust
        self.name = name
        self.cluster_stat = None
        self.vec = np.zeros(self.n_clust)
        self.gt = None
        self.pr = None
        self.features = None

    def hard_stat(self, labels):
        # logger.debug('.')
        self.cluster_stat = np.bincount(labels, minlength=self.n_clust)

    def get_vec(self):
        return self.cluster_stat

    def soft_stat(self, features, centers):
        # logger.debug('.')
        dist = np.sum(centers ** 2, axis=1) + \
               np.sum(features ** 2, axis=1)[:, np.newaxis] - \
               2 * np.dot(features, centers.T)
        self.features = dist

        soft_weights = np.max(dist, axis=1).reshape((-1, 1)) - dist
        soft_weights = soft_weights / np.sum(soft_weights, axis=1).reshape((-1, 1))
        self.cluster_stat = np.sum(soft_weights, axis=0)
        self.cluster_stat = np.nan_to_num(self.cluster_stat)

        # softmax = dist - np.max(dist, axis=1).reshape((-1, 1))
        # softmax = np.exp(-dist)
        # softmax = softmax / np.sum(softmax, axis=1).reshape((-1, 1))
        # self.cluster_stat = np.sum(softmax, axis=0)
        # self.cluster_stat = np.nan_to_num(self.cluster_stat)
