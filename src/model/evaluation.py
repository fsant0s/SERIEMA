import os
import numpy as np
import json
import logging
import math

import ast
from collections import ChainMap

import wandb
from config.definitions import WANDB_API

from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from multiprocessing import Process, Manager

from rpy2.rinterface_lib.embedded import RRuntimeError
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
pandas2ri.activate()

clue = importr("clue")
bootcluster = importr("bootcluster")
OTclust = importr("OTclust")

rng = np.random.RandomState(1)
logger = logging.getLogger(__name__)

def calc_stability_metrics(data, clusters, k_means_random_state, n_samples, randomStateSeed):
    size_sample = data.shape[0]
    stabilities = {}
    
    rData = None
    with localconverter(ro.default_converter + pandas2ri.converter):
        rData = ro.conversion.rpy2py(data)

    for cluster in clusters:
        kmeans = KMeans(n_clusters=cluster, n_init=10, random_state = k_means_random_state)
        labels, indices = get_labels_and_indices(data, kmeans, size_sample, n_samples, randomStateSeed)   

        stabilities["adjusted_rand_score - " + str(cluster)] = adjusted_rand_score_func(labels, indices, cluster)
        stabilities["adjusted_mutual_info_score - " + str(cluster)] = adjusted_mutual_info_score_func(labels, indices, cluster)
        stabilities["bagclust - " + str(cluster)] = bagclust(rData, num_bootstrap_samples = n_samples, n_cluster = cluster)
        stabilities["han - " + str(cluster)] = han(rData, num_bootstrap_samples = n_samples, n_cluster = cluster)
        stabilities["OTstab - " + str(cluster)] = OTstab(rData, num_bootstrap_samples = n_samples, n_cluster = cluster)
    
    return stabilities

def get_labels_and_indices(data, clrt_algorithm, size_sample, n_samples, randomStateSeed):
    rng = np.random.RandomState(randomStateSeed)
    labels = []
    indices = []
    for _ in range(n_samples):
        # draw bootstrap samples, store indices
        sample_indices = rng.randint(0, data.shape[0], size_sample)
        indices.append(sample_indices)
        clrt_algorithm = clone(clrt_algorithm)
        if hasattr(clrt_algorithm, "random_state"):
            # randomize estimator if possible
            clrt_algorithm.random_state = rng.randint(1e5)
        data_bootstrap = data[sample_indices]
        clrt_algorithm.fit(data_bootstrap)
        # store clustering outcome using original indices
        relabel = -np.ones(data.shape[0], dtype=int)
        relabel[sample_indices] = clrt_algorithm.labels_
        labels.append(relabel)
    return (labels, indices)

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
def adjusted_rand_score_func(labels, indices, cluster):
    logger.info("Computing adjusted_rand_score k = "+ str(cluster))
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            in_both = np.intersect1d(i, j)
            scores.append(adjusted_rand_score(l[in_both], k[in_both])) 
    return np.mean(scores)

#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
def adjusted_mutual_info_score_func(labels, indices, cluster):
    logger.info("Computing adjusted_mutual_info_score k = " + str(cluster))
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            in_both = np.intersect1d(i, j)
            scores.append(adjusted_mutual_info_score(l[in_both], k[in_both]))
    return np.mean(scores)

#"A prediction-based resampling method for estimating the number of clusters in a dataset."
#"Bagging to improve the accuracy of a clustering procedure."
#Explicação: Stability estimation for unsupervised clustering: A review
def bagclust(rData, num_bootstrap_samples, n_cluster):
    logger.info("Computing [bagclust]")
    rDataStab = clue.cl_bag(x = rData, B = num_bootstrap_samples, k = n_cluster)
    return rDataStab.rx2['.Data'].max(axis = 1).mean()

#Bootstrapping estimates of stability for clusters,observations and model selection
#Para entender vá para a página 4 Seção 2 Fig. 1.
def han(rData, num_bootstrap_samples, n_cluster):
    logger.info("Computing [han]")
    try:
        hanStab = bootcluster.stability(x = rData, k = n_cluster, B = num_bootstrap_samples)
        stab_overall = float(0) if math.isnan(float(hanStab.rx2['overall'])) else float(hanStab.rx2['overall'])
        return stab_overall
    except RRuntimeError:
        return float(0)

#CPS Analysis for cluster validation
#Install from github https://github.com/cran/OTclust
#Melhor explicação: Denoising Methods for Inferring Microbiome Community Content and Abundance
def OTstab(rData, num_bootstrap_samples, n_cluster): 
    logger.info("Computing [OTstab]")
    otclust = OTclust.clustCPS(rData, k=n_cluster, l= False, pre=False, noi="after",
                nPCA = 2, nEXP = num_bootstrap_samples)
    return float(otclust.rx2['tight_all'])