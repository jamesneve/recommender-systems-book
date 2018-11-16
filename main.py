# -*- coding: UTF-8 -*-

from datasets.loader import *
from chapter2.neighborhoodpearsoncosine import *
from chapter2.clustering import *
import numpy as np

# print("Example matrix")
# example_matrix = [[7, 6, 7, 4, 5, 4], [6, 7, np.NaN, 4, 3, 4], [np.NaN, 3, 3, 1, 1, np.NaN], [1, 2, 2, 3, 3, 4],
#                   [1, np.NaN, 1, 2, 3, 3]]
# print(example_matrix)
#
# pearson_coefficients = calculate_pearson_coefficients(example_matrix)
# print("Pearson Coefficients")
# print(pearson_coefficients)
#
# pearson_rating = predict_pearson_rating((2, 0), example_matrix, pearson_coefficients, 2)
# print("Predicted rating for User 3, Item 1 using Pearson Correlation Coefficient: %f" % pearson_rating)
#
# cosine_coefficients = calculate_adjusted_cosine_coefficients(example_matrix)
# cosine_rating = predict_cosine_rating((2, 0), example_matrix, cosine_coefficients, 2)
# print("Predicted rating for User 3, Item 1 using Pearson Correlation Coefficient %f" % cosine_rating)
#
# print("---------------")
# print("Dimensionality Reduction Step")
#
# print("Estimate missing values")
# rf = compute_rf(example_matrix)
# s = compute_s(rf)
#
# print("Positive semi-definite matrix between pairs of items")
# print(s)
#
# pd = compute_pd(s, 2)
#
# res = np.matmul(rf, pd)
# print("Dimensionality reduced result")
# print(res)

example_matrix = readMovieLens()
print("Dimensionality Reduction Step")

print("Estimate missing values")
rf = compute_rf(example_matrix)
s = compute_s(rf)

print("Positive semi-definite matrix between pairs of items")
print(s)

pd = compute_pd(s, 2)
print("Largest eigenvectors for pd")
print(pd)

res = np.matmul(rf, pd)
print("Dimensionality reduced result")
print(res)