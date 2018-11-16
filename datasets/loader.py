# -*- coding: UTF-8 -*-

import pandas as pd

mlsmall = "datasets/ml-latest-small/ratings.csv"


def readMovieLens():
    df = pd.read_csv(mlsmall)
    df = df.drop(['timestamp'], axis=1)
    res = df.pivot(index='userId', columns='movieId', values='rating')
    matrix = res.values
    return matrix
