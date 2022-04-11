#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 08:00:11 2022

@author: rahelmizrahi
"""

# Python program to create sparse matrices

# Import required package
from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
sizes = [
    (1000, 1000), 
    (1024, 1024),
    (2048, 1024)
    ]
rng = default_rng()
nnz = 0.6 # percentage of tot elems that are non-zero
rvs = stats.poisson(25, loc=10).rvs

for i in range(0, len(sizes)):
    path = "dataset/R" + str(i)
    with open(path, "w+") as f:
        rows = sizes[i][0]
        cols = sizes[i][1]
        S = random(rows, cols, density=nnz, random_state=rng, data_rvs=rvs)
        for line in S:
            f.write(line)
            


