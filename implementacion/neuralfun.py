import numpy as np
from numpy import log, exp

# Logsig activation function -----------------------------------------

def logsig(x):
    
    return 1 / (1 + exp(-x));

def ilogsig(x):
    
    return -log((1/x)-1);

def dlogsig(x):

    return 1/((1+exp(-x))**2)*exp(-x);

# Linear activation function -----------------------------------------

def linear(x):

    return x;

def ilinear(x):
    
    return x;

def dlinear(x):

    return np.ones(len(x));

# ReLu activation function -----------------------------------------

def relu(x):
    
    return log(1+exp(x));

def irelu(x):  # El x debe tener valores > 0 porque es el rango de salida de la función ReLu

    return log(exp(x)-1);

def drelu(x):
  
    return 1 / (1 + exp(-x)); # It is the logistic function