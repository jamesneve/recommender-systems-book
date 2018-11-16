# -*- coding: UTF-8 -*-

from .general import *
import numpy as np
from scipy import linalg


def compute_rf(ur_matrix):
    means = calculate_means(ur_matrix)
    fl_ur_matrix = np.zeros((len(ur_matrix), len(ur_matrix[0])))
    for i, row in enumerate(ur_matrix):
        for j, val in enumerate(row):
            if math.isnan(val):
                fl_ur_matrix[i][j] = means[i]
            else:
                fl_ur_matrix[i][j] = float(ur_matrix[i][j])

    return fl_ur_matrix


def compute_s(rf):
    rft = np.transpose(rf)
    s = np.matmul(rft, rf)
    return s


def compute_pd(s, d):
    eigenvalues, eigenvectors = linalg.eigh(s)
    idx = eigenvalues.argsort()[-d:][::-1]

    eigenvectors = eigenvectors[:, idx]

    return eigenvectors
