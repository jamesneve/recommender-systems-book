# -*- coding: UTF-8 -*-

import math


def calculate_means(ur_matrix):
    means = []
    for row in ur_matrix:
        sum = 0
        cnt = 0
        for val in row:
            if math.isnan(val):
                continue
            sum += val
            cnt += 1
        mean = float(sum) / float(cnt)
        means.append(mean)

    return means
