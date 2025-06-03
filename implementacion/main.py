import numpy as np
import multiprocessing as mp
import sys
from sklearn.metrics import roc_curve, auc
import time
import statistics
from os.path import dirname, join as pjoin
import io
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import random
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.io

# librerías propias
from DAEF import *
from aux_functions import *
from auxiliar import *

#suppress warnings
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    # Cargar y preprocesar el conjunto de datos
    path_cardio = "C:/Users/DAVID/Desktop/implementacion/"  
    mat = scipy.io.loadmat(path_cardio + 'cardio.mat')
    X = pd.DataFrame(mat['X'])
    X.insert(X.shape[1], "Class", pd.DataFrame(mat['y'].flatten()))
    X = X.drop_duplicates()
    X.dropna(axis='columns')
    X = X.reset_index(drop = True)
    Y = pd.Series(X.loc[:, "Class"])
    X = X.iloc[:, 0:X.shape[1]-1]
    X = X.loc[:, (X != X.iloc[0]).any()]
    
    # Normalizar el conjunto de datos
    scaler = train_normalizer(X)
    X = normalize_data(X, scaler)
    
    # Splitear el conjunto de datos 
    normal_data_indexes = Y.index[Y == 0].tolist()
    anomaly_data_indexes = Y.index[Y == 1].tolist()
    random.shuffle(normal_data_indexes)
    random.shuffle(anomaly_data_indexes)
    train_normal_data_indexes, test_normal_data_indexes = train_test_split(normal_data_indexes, test_size=0.1, random_state=17)        
    anomaly_data_indexes = anomaly_data_indexes[0:len(test_normal_data_indexes)] 
    test_normal_data_indexes = test_normal_data_indexes[0:len(anomaly_data_indexes)] 
    X_train = X.iloc[train_normal_data_indexes, :].to_numpy()
    Y_train = Y.iloc[train_normal_data_indexes].to_numpy()
    X_test = X.iloc[test_normal_data_indexes + anomaly_data_indexes, :].to_numpy()
    Y_test = Y.iloc[test_normal_data_indexes + anomaly_data_indexes].to_numpy()
    X_train = X_train.T
    X_test = X_test.T
        
    # Parámetros del método
    lam_HL = 0.9
    lam_LL = 0.9
    procesos = 2
    arquitectura = [X.shape[1], 4, 8, 12, 16, X.shape[1]] # [21, 4, 8, 12, 16, 21]
    
    # Inicializar pesos y bias
    pesos_list, bias_list = init_net_parameters(arquitectura, "xavier", 2)
    # Entrenar con un subconjunto de datos
    _, net_parameters1, local_matrices1 = DAEF_train_locally(X_train[:, 0:700], pesos_list, bias_list, arquitectura, lam_HL, lam_LL, procesos)
    
    # Entrenar con otro subconjunto de datos
    _, net_parameters2, local_matrices2 = DAEF_train_locally(X_train[:, 700:], pesos_list, bias_list, arquitectura, lam_HL, lam_LL, procesos)
    
    # Agregarlos 
    net_parameters3, final_matrices3 = DAEF_train_incrementally(local_matrices1, local_matrices2, arquitectura, lam_HL, lam_LL, procesos)
    
    # Entrenar con un el conjunto de datos completo equivalente a la suma de los dos subconjuntos
    _, net_parameters4, final_matrices4 = DAEF_train_locally(X_train, pesos_list, bias_list, arquitectura, lam_HL, lam_LL, procesos)
    
    # Evaluar el modelo incremental sobre el conjunto de test
    prediction1train = DAEF_predict(X_train, net_parameters3)
    error1 = compute_train_error(X_train, prediction1train, 90)
    print("Versión incremental: ")
    clasification = predict_and_clasify (X_test, Y_test, net_parameters3, error1)
    
    # Evaluar el modelo que ha entrenado con todos los datos sobre el conjunto de test
    prediction2train = DAEF_predict(X_train, net_parameters4)
    error2 = compute_train_error(X_train, prediction2train, 90)
    print("Versión NO incremental: ")
    predict_and_clasify (X_test, Y_test, net_parameters4, error2)