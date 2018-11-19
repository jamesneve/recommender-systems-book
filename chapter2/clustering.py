# -*- coding: UTF-8 -*-

import random
import numpy as np
from scipy.spatial.distance import cdist, squareform


def get_representatives(mat, n):
    samples = random.sample(range(0, len(mat)), n)

    sample_hash = {}
    for s in samples:
        sample_hash[s] = mat[s]

    return sample_hash


def cluster_around_representatives(mat, sample_hash):
    clusters_map = {}
    for k, v in sample_hash.items():
        clusters_map[k] = [k]

    for i, row in enumerate(mat):
        if i in sample_hash:
            continue

        min_edistance = []
        for k, v in sample_hash.items():
            edistance = np.sqrt(np.nansum((np.array(row) - np.array(v))**2))
            if len(min_edistance) == 0 or min_edistance[1] > edistance:
                min_edistance = [k, edistance]

        clusters_map[min_edistance[0]].append(i)

    return clusters_map


def matrices_from_clusters(matrix, clusters_map):
    matrices_list = []
    for k, v in clusters_map.items():
        mat = []
        for rownum in v:
            mat.append(matrix[rownum])
        matrices_list.append(mat)

    return matrices_list
