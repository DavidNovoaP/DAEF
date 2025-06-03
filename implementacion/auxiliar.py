from OL_reg import *
import numpy as np
import multiprocessing as mp
import time
from split_optim import split_optim
from dsvd import dsvd
from auxiliar import *
from scipy.sparse import csr_matrix, diags
from neuralfun import logsig, ilogsig, dlogsig, relu, irelu, drelu, linear, ilinear, dlinear
import sys
from scipy.linalg import orth
from pathos.multiprocessing import ProcessingPool as Pool


# Calculo paralelizado de H
def func1(args):
    fh = args[0]
    W1 = args[1]
    Xs = args[2]
    H = eval(fh)(W1.transpose() @ Xs)
    return H
 
# Cálculo paralelizado de HF y M  
def func2(args): 
    j = args[0]
    finv = args[1]
    Xs = args[2]
    fderiv = args[3]
    H = args[4]
    
    f_d = eval(finv)(Xs[j,:]) 
    f_d = f_d.transpose()
    f  = eval(fderiv)(f_d) 
    
    auxiliar = csr_matrix(f).toarray()
    auxiliar = auxiliar[0]
    F = np.diag(auxiliar)
    HF = H @ F
    M  = H @ (f * (f * f_d))
    return HF, M
      
# Calculo paralelizado de salidas    
def func3(args):    
    fn = args[0]
    W2 = args[1]
    H = args[2]
    
    salidasTr = eval(fn)(W2.transpose() @ H)
    return salidasTr

def calcularPesos1CapaEncoder (Xs, NH, fh, procs, process_pool):
    # Pesos de la primera capa
    Up, Sp = dsvd(Xs,len(Xs),procs, process_pool);
      
    # Salidas de la primera capa
    W1 = Up[:,0:NH];
    args = []; 
    for i in range(len(Xs)):
        args.append((fh,W1,Xs[i]));
    H = list(process_pool.imap(func1, args))
    H = np.hstack(H)
    return H, W1, Up, Sp
    
      
    
def calcularPesos1CapaDecoder (X1, fh, finv, fderiv, lam, process_pool, W1, B1):
    
    samples, features  = X1.shape

    # Salida de la capa oculta del AE auxiliar (H1)
    H1 = eval(fh)(X1 @ W1 + B1)
    
    # Aplicamos ROLANN para construir el decoder del AE auxiliar
    # Un modelo por cada neurona de la capa de salida
    # Version distribuida
    
    arguments_iterable = []
    for i in range (0, features):
        arguments_iterable.append((H1.T, X1[:, i], finv, fderiv, lam, None, None, None))
    args = list(process_pool.imap(onelayer_reg, arguments_iterable)) 
    W2 = [item[0] for item in args]
    W2 = np.array(W2)
    W2 = np.vstack(W2)
    
    M = [item[1] for item in args]    
    U = [item[2] for item in args]
    S = [item[3] for item in args]
    
    Hi = eval(fh)(W2.T @ X1.T)
    return Hi, W2, M, U, S
    
    
def calcularPesos1CapaEncoderIncremental (NH, US_antes, US_nueva):
    
    # Pesos de la primera capa
    U_antes, S_antes = US_antes
    U_nueva, S_nueva = US_nueva
    
    Up, Sp, Vp = np.linalg.svd(np.concatenate([U_antes@S_antes, U_nueva@S_nueva], axis=1), full_matrices = False); # economy
    Sp = np.diag(Sp);
    
    # Salidas de la primera capa
    W1 = Up[:,0:NH];
    
    return W1, Up, Sp
    
    
def calcularPesos1CapaDecoderIncremental (MUS_antes, MUS_nueva, lam, process_pool):
    Mk, Uk, S_k_sparse = MUS_antes
    Mp, Up, S_p_sparse = MUS_nueva
    args = []
    args_iterable = []
    for i in range (0, len(Mp)):
        S_p_sparse_aux = sp.sparse.spdiags(S_p_sparse[i], 0, S_p_sparse[i].size, S_p_sparse[i].size, format = "csr")
        S_k_sparse_aux = sp.sparse.spdiags(S_k_sparse[i], 0, S_k_sparse[i].size, S_k_sparse[i].size, format = "csr")
        args_iterable.append((Mp[i], Up[i], S_p_sparse_aux, lam, Mk[i], Uk[i], S_k_sparse_aux))
    args = list(process_pool.imap(onelayer_reg_sin_datos_X, args_iterable)) 
      
    
    W2 = [item[0] for item in args]
    W2 = np.array(W2)
    W2 = np.vstack(W2)

    M = [item[1] for item in args]    
    U = [item[2] for item in args]
    S = [item[3] for item in args]
    return W2, M, U, S
    
 
    
    
    
    
    
    
    
    
    
    
    
    