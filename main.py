# -*- coding: UTF-8 -*-

from datasets.loader import *
from chapter2.neighborhoodpearsoncosine import *
from chapter2.clustering import *
from chapter3.latentfactorsgd import *
import numpy as np

print("Example matrix")
# example_matrix = [[7, 6, 7, 4, 5, 4], [6, 7, np.NaN, 4, 3, 4], [np.NaN, 3, 3, 1, 1, np.NaN], [1, 2, 2, 3, 3, 4],
#                   [1, np.NaN, 1, 2, 3, 3]]
# example_matrix = np.array([
#     [5, 3, 0, 1],
#     [4, 0, 0, 1],
#     [1, 1, 0, 5],
#     [1, 0, 0, 4],
#     [0, 1, 5, 4],
# ])
# print(example_matrix_2)

example_matrix = readMovieLens()

lfgd = LatentFactorSGD(example_matrix, 10, 0.01, 30, 0.1)
lfgd.uniform_initialize_uv()
u, v = lfgd.train()
res = np.dot(u, v.T)

# print("---- Dimensionality Reduction ---")
# print("Estimate missing values")
# rf = compute_rf(example_matrix)
# s = compute_s(rf)
#
# print("Generate positive semi-definite matrix between pairs of items")
#
# pd = compute_pd(s, 2)
# print("Find largest eigenvectors for pd")
#
# res = np.matmul(rf, pd)
#
# print("--- Clustering ---")
# print("Get representatives from eigenvector matrix")
# reps = get_representatives(res, 30)
#
# print("Cluster")
# clusters = cluster_around_representatives(res, reps)
#
# print("Get matrices from clusters")
# matrices = matrices_from_clusters(rf, clusters)
#
#
# print("-- Collaborative Filtering --")
# print("Here's one of the clusters - because of sampling, it'll be different every time")
# example_cluster = matrices[3]
# print(example_cluster)

# print("Get adjusted cosine coefficients for one of the clusters")
# cosine_coefficients = calculate_adjusted_cosine_coefficients(example_cluster)
#
# print("Predict rating for user X, item X")
# cosine_rating = predict_cosine_rating((2, 0), example_cluster, cosine_coefficients, 2)
#
# print("Cosine rating result")
# print(cosine_rating)

# print("Get pearson coefficients for one of the clusters")
# pearson_coefficients = calculate_pearson_coefficients(example_cluster)
#
# print("Predict rating for user 2, item 0")
# pearson_rating = predict_pearson_rating((2, 0), example_cluster, pearson_coefficients, 2)
#
# print("Predicted rating for user 2, item 0 ")
# print(pearson_rating)
