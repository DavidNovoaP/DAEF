# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:27:57 2021

@author: usuario
"""

import numpy as np
import math
from scipy.linalg import toeplitz
from numpy import linalg as LA

# create the positive definite matrix P
p = 3
A = np.random.rand(p,p)
P = (A + np.transpose(A))/2 + p*np.eye(p)

# get the subset of its eigenvectors 
vals, vecs = LA.eig(P)
w = vecs[:,0:p]

#check that it's orthogonal 
print("matrix shape: ",w.shape)
print("check orthogonality: ",np.matmul(np.transpose(w),w))
print("has complex elements: ",np.max(np.iscomplex(w)))