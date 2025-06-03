# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 11:18:48 2020

@author: DAVID
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#from svdd import *

import h2o
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator

from aux_functions import *

# #####################################################################

def sk_classify (data, model):
    print("Classifying...")
    y_pred = model.predict(data)
    print("Classified.")
    return pd.Series(y_pred).apply(change_target_value_01)

# #####################################################################
# ONE CLASS SUPPORT VECTOR MACHINE
    
def OCSVM_train (data, nu, kernel, gamma):
    print("OC-SVM trainning...")
    model = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(data)
    print("OC-SVM trained.")
    return model

# #####################################################################
# ROBUST COVARIANCE
    
def RC_train (data, contamination):
    print("RC trainning...")
    model = EllipticEnvelope(contamination=contamination)
    model.fit(data)
    print("RC trained.")
    return model

# #####################################################################
# ISOLATION FOREST
    
def IF_train (data, n_estimators, contamination, random_state):
    print("IF trainning...")
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
    model.fit(data)
    print("IF trained.")
    return model

# #####################################################################
# LOCAL OUTLIER FACTOR
    
def LOF_train (data, n_neighbors, contamination, novelty, algorithm):
    print("LOF trainning...")
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=novelty, algorithm=algorithm)
    model.fit(data)
    print("LOF trained.")
    return model

# #####################################################################
# AUTOENCODER

def AE_train (data, hidden, epochs):
    print("AE trainning...")
    h2o.init(port = 54321, min_mem_size_GB=8)
    dataH2O = h2o.H2OFrame(data)
    
    model = H2OAutoEncoderEstimator(
        activation="tanh",
        hidden=hidden,
        sparse=True,
        reproducible=True,
        input_dropout_ratio=0,
        seed=134,
        standardize=False,
        ignore_const_cols=False,
        epochs=epochs)
    
    model.train(x=dataH2O.names, training_frame=dataH2O)
    
    recon_error_test = model.anomaly(dataH2O)
    recon_error_test = recon_error_test.sort(0, ascending=False)
    
    #indiceUmbral = round((recon_error_test.dim[0]*sensivity_threshold))
    #margenExtremadamenteAtipico = recon_error_test[indiceUmbral, 0]
    
    print("AE trained.")
    return model, recon_error_test

def AE_classify (data, model_threshold):
    print("Classifying...")
    model, threshold = model_threshold
    h2o.init(port = 54321)
    dataH2O = h2o.H2OFrame(data)
    error = model.anomaly(dataH2O).as_data_frame(use_pandas  = True).to_numpy()
    y_pred = np.where(error < threshold, 0, 1)
    print("Classified.")
    return  pd.Series(y_pred.flatten())

# #####################################################################
# DEEP SUPPORT VECTOR DATA DESCRIPTION
""" 
def SVDD_train (trainX, trainY, positive_penalty, negative_penalty, kernel):
    kernelList = {"1": {"type": 'gauss', "width": 1/24},
              "2": {"type": 'linear', "offset": 0},
              "3": {"type": 'ploy', "degree": 2, "offset": 0},
              "4": {"type": 'tanh', "gamma": 1e-4, "offset": 0},
              "5": {"type": 'lapl', "width": 1/12}
              }
    
    parameters = {"positive penalty": positive_penalty,
                  "negative penalty": negative_penalty,
                  "kernel": kernelList[kernel],
                  "option": {"display": 'off'}}
    
    print("SVDD trainning...")
    svdd = SVDD(parameters)
    svdd.train(trainX, trainY)
    print("SVDD trained.")
    return svdd

def SVDD_classify (testData, testLabel, model):
    print("Classifying...")
    _, _, predictedlabel = model.test(testData, testLabel)
    print("Classified.")
    return pd.Series(pd.DataFrame(predictedlabel).to_numpy().flatten()).apply(change_target_value_01)


"""


