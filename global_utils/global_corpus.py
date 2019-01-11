#!/usr/bin/env python

"""Load all features for several classes (entire dataset) and train joined
embedding to discriminate between classes and then apply the segmentation
algorithm.
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'


import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering


from utils.logging_setup import logger
from utils.arg_pars import opt
from utils.util_functions import join_data, timing, join_return_stat, parse_return_stat
from utils.mapping import GroundTruth
from models.dataset_loader import load_reltime
from models import mlp
from models.training_embed import training, load_model
from eval_utils.accuracy_class import Accuracy
from eval_utils.f1_score import F1Score
from corpus import Corpus
from global_utils.action_video import ActionVideo
from global_utils.local_corpus import LocalCorpus
import BF_utils.update_argpars as bf_utils
import YTI_utils.update_argpars as yti_utils


class GlobalCorpus:
    @timing
    def __init__(self, actions, K):
        logger.debug('.')
        self._actions = actions
        self._video2idx = {}
        self._idx2video = {}
        self._assemblage = {}
        self._features_embed = None
        self._long_gt = []
        self._gl_action_labels = None
        self._action_videos = []
        self._actions_idxs = []
        self._video_idxs = []
        self._K = K
        for action in actions:
            opt.subaction = action
            corpus = Corpus(Q=opt.gmm, subaction=action, K=K)
            self._assemblage[action] = corpus

            for video_idx, video in enumerate(corpus.get_videos()):
                self._video2idx[video.name] = video_idx
        # shuffle indexes since the video loaded by actions
        shuffle_idx = np.arange(len(self._video2idx))
        np.random.shuffle(shuffle_idx)
        for idx, video_name in enumerate(self._video2idx):
            video_idx = shuffle_idx[idx]
            self._video2idx[video_name] = video_idx
            self._idx2video[video_idx] = video_name

    @timing
    def train_mlp(self):
        torch.manual_seed(opt.seed)
        dataloader = load_reltime(assemblage=self._assemblage,
                                  actions=self._actions)

        model, loss, optimizer = mlp.create_model()  # regression

        if opt.resume:
            model.load_state_dict(load_model(name=opt.model_name))
            self._embedding = model
        else:
            self._embedding = training(dataloader,
                                       opt.epochs,
                                       save=opt.save_model,
                                       model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       name=opt.model_name)

        self._embedding = self._embedding.cpu()

        for action_idx, action in enumerate(self._actions):
            features_embed = torch.Tensor(self._assemblage[action].get_features())
            features_embed = self._embedding.embedded(features_embed).detach().numpy()

            self._features_embed = join_data(self._features_embed,
                                             features_embed,
                                             np.vstack)
            self._long_gt += list([action_idx] * features_embed.shape[0])

    @timing
    def clustering(self, type='kmeans_batch', n_clust=50):
        # todo: does it make sense to try other clustering technique here? why I apply everywhere k-means, fuck & shit
        logger.debug('# of clusters: %d type: %s' % (n_clust, type))
        np.random.seed(opt.seed)

        if type == 'kmeans':
            self.kmeans = KMeans(n_clusters=n_clust, random_state=opt.seed,
                                 n_jobs=-1)
        if type == 'kmeans_batch':
            batch_size = 100 if opt.reduced else 30
            self.kmeans = MiniBatchKMeans(n_clusters=n_clust,
                                          random_state=opt.seed,
                                          batch_size=batch_size)

        # todo: what does it mean??
        logger.debug(str(self.clustering))

        logger.debug('Shape %d' % self._features_embed.shape[0])
        mask = np.zeros(self._features_embed.shape[0], dtype=bool)
        step = 1
        for i in range(0, mask.shape[0], step):
            mask[i] = True

        self.kmeans.fit(self._features_embed[mask])

        accuracy = Accuracy()
        accuracy.predicted_labels = self.kmeans.labels_
        accuracy.gt_labels = np.asarray(self._long_gt)[mask]

        accuracy.mof()
        logger.debug('MoF val: ' + str(accuracy.mof_val()))

    def cluster_stat(self, n_clust):
        np.random.seed(opt.seed)
        long_gt = []
        long_features = None
        for action_idx, action in enumerate(self._actions):
            features = self._assemblage[action].get_features()
            features = torch.Tensor(features)
            features = self._embedding.embedded(features).detach().numpy()
            predicted_labels = self.kmeans.predict(features)
            for video_idx, video in enumerate(self._assemblage[action].get_videos()):
                video.update_z(predicted_labels[video.global_range])
                action_video = ActionVideo(name=video.name, n_clust=n_clust)

                action_video.soft_stat(features[video.global_range], self.kmeans.cluster_centers_)
                # action_video.hard_stat(predicted_labels[video.global_range])

                self._video_idxs.append(video_idx)
                self._actions_idxs.append(action_idx)
                long_gt.append(action_idx)
                long_features = join_data(long_features,
                                          action_video.get_vec(),
                                          np.vstack)

        # video level classification
        kmeans = KMeans(n_clusters=len(self._actions), random_state=opt.seed)
        # kmeans = KMeans(n_clusters=5, random_state=opt.seed)
        logger.debug(str(kmeans))
        kmeans.fit(long_features)

        accuracy = Accuracy()
        accuracy.predicted_labels = kmeans.labels_
        accuracy.gt_labels = long_gt
        accuracy.mof()
        accuracy.mof_classes()
        logger.debug('MoF val: ' + str(accuracy.mof_val()))

        self._gl_action_labels = kmeans.labels_

    def segmentation(self, epochs=10, lr=1e-3, dim=30):
        long_gt = []
        long_pr = []
        pr_idx_start = 0
        return_stat_all = None
        n_videos = 0
        for pr_action_idx, pr_action in enumerate(np.unique(self._gl_action_labels)):
            logger.debug('\n\nAction # %d, label %d' % (pr_action_idx, pr_action))
            mask = self._gl_action_labels == pr_action
            pr_features = None
            pr_videos = []
            gt_actions_idxs = np.asarray(self._actions_idxs)[mask]
            video_idxs = np.asarray(self._video_idxs)[mask]
            global_start = 0
            for gt_action_idx in np.unique(gt_actions_idxs):
                mask_corpus = gt_action_idx == gt_actions_idxs
                gt_action = self._actions[gt_action_idx]

                video_idxs_corpus = video_idxs[mask_corpus]
                videos = self._assemblage[gt_action].video_byidx(video_idxs_corpus)
                feature_mask = np.zeros(videos[0].global_range.shape[0], dtype=bool)
                for video in videos:
                    feature_mask += video.global_range
                    # changing video indexing in the corresponding corpus
                    video.global_start = global_start
                    global_start += video.n_frames

                pr_features = join_data(pr_features,
                                        self._assemblage[gt_action].get_features()[feature_mask],
                                        np.vstack)
                pr_videos += list(videos)

            total = pr_features.shape[0]
            for video in pr_videos:
                video.update_indexes(total)

            # for each predicted subaction class there it's own embedding training
            # corpus = LocalCorpus(pr_videos, pr_features, K=self._K)
            # for all local pipelines there is the same already trained embedding
            corpus = LocalCorpus(pr_videos, pr_features, K=self._K, embedding=self._embedding)
            corpus.pipeline(epochs=epochs, lr=lr, dim=dim)
            corpus_pr, corpus_gt = corpus.pr_gt(pr_idx_start)
            return_stat_single = corpus.stat()
            return_stat_all = join_return_stat(return_stat_all, return_stat_single)
            pr_idx_start += self._K

            long_pr += corpus_pr
            long_gt += corpus_gt

            n_videos += len(corpus)

        parse_return_stat(return_stat_all)

        self.accuracy_corpus(long_pr, long_gt, n_videos, prefix='Final')

    def accuracy_corpus(self, prediction, ground_truth, n_videos=None, prefix=''):
        accuracy = Accuracy()
        accuracy.predicted_labels = prediction
        accuracy.gt_labels = ground_truth
        if opt.bg:
            # enforce bg class to be bg class
            accuracy.exclude[-1] = [-1]
        accuracy.mof()
        accuracy.mof_classes()
        accuracy.iou_classes()
        logger.debug('%s MoF val: ' % prefix + str(accuracy.mof_val()))

        if n_videos is not None:
            f1_score = F1Score(K=self._K * len(self._actions), n_videos=n_videos)
            f1_score.set_gt(ground_truth)
            f1_score.set_pr(prediction)
            f1_score.set_gt2pr(accuracy._gt2cluster)
            if opt.bg:
                f1_score.set_exclude(-1)
            f1_score.f1()

    def baseline_kmeans(self):
        logger.debug('.')
        features = None
        long_gt = []
        for action_idx, action in enumerate(self._actions):
            for video in self._assemblage[action].get_videos():
                long_gt += list(video.gt)

            if self._features_embed is None:
                features = join_data(features,
                                     self._assemblage[action].get_features(),
                                     np.vstack)

        if self._features_embed is not None:
            features = self._features_embed

        assert len(long_gt) == features.shape[0]
        idxs = np.arange(features.shape[0])
        np.random.shuffle(idxs)

        features = features[idxs]
        long_gt = np.asarray(long_gt)[idxs]

        kmeans = MiniBatchKMeans(n_clusters=50,
                                 random_state=opt.seed,
                                 batch_size=200)

        # kmeans = KMeans(n_clusters=48, random_state=opt.seed)

        kmeans.fit(features)

        self.accuracy_corpus(prediction=kmeans.labels_,
                             ground_truth=np.asarray(long_gt),
                             prefix='Baseline clusering')


if __name__ == '__main__':
    if opt.dataset == 'bf':
        bf_utils.update()
    if opt.dataset == 'yti':
        yti_utils.update()
    # '10cl.joined' '10cl.relt.!idx'
    actions = ['coffee', 'cereals', 'milk', 'tea', 'juice', 'sandwich', 'salat', 'friedegg', 'scrambledegg', 'pancake']
    # actions = ['coffee', 'cereals', 'friedegg', 'scrambledegg']
    # actions = ['coffee', 'cereals', 'milk', 'tea']
    actions = ['coffee', 'cereals']
    # opt.reduced = 30

    # actions = ['cpr', 'changing_tire', 'coffee', 'jump_car', 'repot']
    # actions = ['cpr', 'coffee', 'repot']
    # actions = ['cpr', 'coffee']

    # opt.ordering = False
    # opt.viterbi = True
    # opt.rt_concat = 0
    K = 5
    joined_corpus = GlobalCorpus(actions=actions, K=K)

    joined_corpus.train_mlp()

    # joined_corpus.baseline_kmeans()
    # exit(0)

    n_clust = 50
    # n_clust = len(actions)
    # for n_clust in [8, 16, 32, 35, 40, 45, 50, 64]:
    # logger.debug('\n\nSET: n_clust: %d\n' % n_clust)
    joined_corpus.clustering(n_clust=n_clust)
    joined_corpus.cluster_stat(n_clust=n_clust)

    epochs = opt.local_epoch
    lr = opt.local_lr
    dim = opt.local_dim
    # for epoch in [30, 60, 90]:
    logger.debug('SET: reduced: %s\tordering: %s\tviterbi: %s\tK: %d\tepochs: %d\tlr: %.1e\tdim: %d' %
                 (str(opt.reduced), str(opt.ordering), str(opt.viterbi), K, epochs, lr, dim))
    joined_corpus.segmentation(epochs=epochs, lr=lr, dim=dim)

