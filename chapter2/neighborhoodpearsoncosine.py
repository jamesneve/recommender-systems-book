# -*- coding: UTF-8 -*-

from .dimensionalityreduction import *
import numpy as np


def calculate_mean_centred_matrix(ur_matrix):
    means = calculate_means(ur_matrix)
    mean_centred_matrix = []
    for i, row in enumerate(ur_matrix):
        mean_centred_matrix.append([])
        for j in row:
            if math.isnan(j):
                mean_centred_matrix[i].append(np.NaN)
                continue
            mean_centred_matrix[i].append(j - means[i])
    return mean_centred_matrix


def calculate_pearson_coefficients(ur_matrix):
    mean_centred_matrix = calculate_mean_centred_matrix(ur_matrix)
    pearson_coefficients = np.zeros((len(ur_matrix), len(ur_matrix)))

    for i in range(0, len(ur_matrix)):
        for j in range(i, len(ur_matrix)):
            if i == j:
                pearson_coefficients[i][j] = 1.0
                continue
            coeff = pearson_coefficient(mean_centred_matrix[i], mean_centred_matrix[j])
            pearson_coefficients[i][j] = pearson_coefficients[j][i] = coeff

    return pearson_coefficients


def pearson_coefficient(su, sv):
    num = 0
    denu = 0
    denv = 0
    for i in range(0, len(su)):
        if math.isnan(su[i]) or math.isnan(sv[i]):
            continue
        num += su[i] * sv[i]
        denu += su[i] ** 2
        denv += sv[i] ** 2

    den = np.sqrt(denu) * np.sqrt(denv)
    return float(num) / den


def calculate_adjusted_cosine_coefficients(ur_matrix):
    mean_centred_matrix = calculate_mean_centred_matrix(ur_matrix)
    cosine_coefficients = np.zeros((len(ur_matrix[0]), len(ur_matrix[0])))

    for i in range(0, len(ur_matrix[0])):
        for j in range(i, len(ur_matrix[0])):
            if i == j:
                cosine_coefficients[i][j] = 1.0
            coeff = cosine_coefficient(mean_centred_matrix, (i, j))
            cosine_coefficients[i][j] = cosine_coefficients[j][i] = coeff

    return cosine_coefficients


def cosine_coefficient(mean_centred_matrix, item_pair):
    num = 0
    denu = 0
    denv = 0
    for row in mean_centred_matrix:
        if math.isnan(row[item_pair[0]]) or math.isnan(row[item_pair[1]]):
            continue
        num += row[item_pair[0]] * row[item_pair[1]]
        denu += row[item_pair[0]] ** 2
        denv += row[item_pair[1]] ** 2

    den = np.sqrt(denu) * np.sqrt(denv)
    return float(num) / den


def predict_pearson_rating(user_rating, ur_matrix, pearson_coefficients, closest_users):
    user_coefficient_pairs = []
    for i, u in enumerate(ur_matrix):
        if i == user_rating[0]:
            continue
        coefficient = pearson_coefficients[i][user_rating[0]]
        user_coefficient_pairs.append((i, coefficient))
    user_coefficient_pairs.sort(key=lambda x: x[1])
    user_coefficient_pairs.reverse()
    user_coefficient_pairs = user_coefficient_pairs[0:closest_users]

    num = 0.0
    den = 0.0
    for p in user_coefficient_pairs:
        num += ur_matrix[p[0]][user_rating[1]] * p[1]
        den += p[1]

    return num / den


def predict_cosine_rating(user_rating, ur_matrix, cosine_coefficients, closest_items):
    item_coefficient_pairs = []
    for i, u in enumerate(ur_matrix[0]):
        if i == user_rating[1]:
            continue
        coefficient = cosine_coefficients[i][user_rating[1]]
        item_coefficient_pairs.append((i, coefficient))
    item_coefficient_pairs.sort(key=lambda x: x[1])
    item_coefficient_pairs.reverse()
    item_coefficient_pairs = item_coefficient_pairs[0:closest_items]

    num = 0.0
    den = 0.0
    for p in item_coefficient_pairs:
        num += ur_matrix[user_rating[0]][p[0]] * p[1]
        den += p[1]

    return num / den
