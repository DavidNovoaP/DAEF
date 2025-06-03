import numpy as np
import scipy as sp
import scipy.sparse
from scipy.sparse import csr_matrix
from neuralfun import logsig, ilogsig, dlogsig, relu, irelu, drelu, linear, ilinear, dlinear

#########################################################################################            
# Syntax:
# ------
# w = onelayer_reg(W,x,f)
#
# Parameters of the function:
# --------------------------
# X : inputs of the network (size: m x n).
# d : desired outputs for the given inputs.
# finv : inverse of the activation function. 
# fderiv: derivative of the activation function.
# lam : regularization term (lambda)
#
# Returns:
# -------
# Optimal weights (w) of the network
    
def onelayer_reg(args):
    import numpy as np
    X,d,finv,fderiv,lam, Mk, Uk, Sk = args 
    # Number of data points (n)
    n = np.size(X,1);
    
    # The bias is included as the first input (first row)
    Xp = np.insert(X, 0, np.ones(n), axis=0);
    
    # Inverse of the neural function
    f_d = eval(finv)(d);
    
    # Derivate of the neural function
    derf = eval(fderiv)(f_d);
    
    F_sparse = sp.sparse.spdiags(derf, 0, derf.size, derf.size, format = "csr")
    Xp_sparse = csr_matrix(Xp)
    H_sparse = Xp_sparse @ F_sparse  
    H = H_sparse.toarray()
    
    
    # If all the input matrices are empty
    if Mk is None and Uk is None and Sk is None:  
        # Modo no incremental
        f_d_sparse = csr_matrix(f_d).T
        M = Xp_sparse @ (F_sparse @ F_sparse @ f_d_sparse)
        M = M.toarray().flatten()
        U, S, _ = np.linalg.svd(H, full_matrices=False);
    elif Mk is not None and Uk is not None and Sk is not None:
        # Modo incremental
        f_d_sparse = csr_matrix(f_d).T
        Mk_sparse = csr_matrix(Mk)
        if Mk_sparse.shape[0] == 1:
            Mk_sparse = Mk_sparse.transpose()
        M = Mk_sparse + Xp_sparse @ (F_sparse @ F_sparse @ f_d_sparse)
        Up, Sp, _ = np.linalg.svd(H, full_matrices=False);  
        Sp_sparse = sp.sparse.spdiags(Sp, 0, Sp.shape[0], Sp.shape[0], format = "csr")
        Up_sparse = csr_matrix(Up)
        aux1 = Up_sparse @ Sp_sparse
        aux1 = aux1.toarray()
        Sk_sparse = sp.sparse.spdiags(Sk, 0, Sk.shape[0], Sk.shape[0], format = "csr")
        aux2 = Uk @ Sk_sparse
        U, S, _ = np.linalg.svd(np.concatenate((aux2, aux1),axis=1), full_matrices=False);        
    else:
        print('Error: All the input matrices (Mk,Uk,Sk) must be all None or all not None');
        return
    
    I_ones = np.ones(np.size(S))
    I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format = "csr")
    S_sparse = sp.sparse.spdiags(S, 0, S.size, S.size, format = "csr")
    
    aux2 = S_sparse * S_sparse + lam * I_sparse
    aux2 = aux2.toarray()
    
    # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
    w = U @ (np.linalg.pinv(aux2) @ (U.transpose() @ M));
    
    args = w, M, U, S_sparse
    
    return args


def onelayer_reg_sin_datos_X(args):
    Mp, Up, S_p_sparse, lam, Mk, Uk, S_k_sparse = args 
    
    if isinstance(Mp, list):
        M = Mk + sum(Mp)
    else:
        M = Mk + Mp
    
    aux2 = Uk @ S_k_sparse
    concatenations_list = [aux2]

    if isinstance(Up, list):
        for i in range (0, len(Up)):
            aux1 = Up[i] @ S_p_sparse[i]
            concatenations_list.append(aux1)
    else:
        aux1 = Up @ S_p_sparse
        concatenations_list.append(aux1)
        
    U, S, _ = np.linalg.svd(np.concatenate(tuple(concatenations_list) ,axis=1), full_matrices=False);        
    I_ones = np.ones(np.size(S))
    I_sparse = sp.sparse.spdiags(I_ones, 0, I_ones.size, I_ones.size, format = "csr")
    S_sparse = sp.sparse.spdiags(S, 0, S.shape[0], S.shape[0], format = "csr")
    
    aux = S_sparse * S_sparse + lam * I_sparse
    aux = aux.toarray()
    
    # Optimal weights: the order of the matrix and vector multiplications has been done to optimize the speed
    w = U @ (np.linalg.pinv(aux) @ (U.transpose() @ M));
    
    args = w, M, U, S_sparse
    
    return args
    
#########################################################################################            
# Syntax:
# ------
# Output = nnsimul(W,x,f)
#
# Parameters of the function:
# --------------------------
# W : weights of the neural network (size: m+1 x 1). The 1st element is the bias.
# X : inputs of the network (matrix of size: m x n).
# f : neural function. 
# 
# Returns:
# -------
# Outputs of the network for all the input data.

def nnsimul(W,X,f):

    # Number of variables (m) and data points (n)
    m,n=X.shape;

    # Neural Network Simulation
    return eval(f)(W.transpose() @ np.insert(X, 0, np.ones(n), axis=0));
                        
#########################################################################################    