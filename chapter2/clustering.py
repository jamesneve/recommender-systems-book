# -*- coding: UTF-8 -*-

import random

def get_representatives(mat, n):
    samples = random.sample(range(0, len(mat)), n)
    return samples
