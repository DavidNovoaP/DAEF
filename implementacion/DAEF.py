import numpy as np
import multiprocessing as mp
import time
from split_optim import split_optim
from dsvd import dsvd
from auxiliar import *
from scipy.sparse import csr_matrix, diags
from neuralfun import logsig, ilogsig, dlogsig, relu, irelu, drelu, linear, ilinear, dlinear
import sys
from pathos.multiprocessing import ProcessingPool as Pool
from aux_functions import *
from auxiliar import *
from sklearn import preprocessing
 
def init_net_parameters (arquitectura, mode, seed):
    import random
    
    if seed != "":
        random.seed(seed)
    W_list = []
    B_list = []
    for i in range (1, len(arquitectura)-2):
        features  = arquitectura[i]
        NH = arquitectura[i+1]
        if mode == "random": # Inicializacion aleatoria
            W1 = np.random.randn(features, NH)
            B1 = np.random.randn(1, NH)
        elif mode == "orthonormal": # Inicializacion Orthonormal
            W1 = np.random.randn(features, NH)
            W1 = orth(W1.T).T # https://github.com/tobifinn/pyextremelm/blob/master/pyextremelm/builder/layers/random.py
            B1 = np.random.randn(1, NH)
            B1 = np.linalg.qr(B1.T)[0].T
        elif mode == "xavier": # Inicializacion Xavier Glorot
            limit = np.sqrt(2 / float(features + NH))
            W1 = np.random.normal(0.0, limit, size=(features, NH))
            B1 = np.random.randn(1, NH)
        W_list.append(W1)
        B_list.append(B1)
    return W_list, B_list

def DAEF_train_locally(X, W_list, B_list, arquitectura, lam_HL, lam_LL, procesos):
    # Entrenamiento local 
    
    fn = 'linear'
    fh = 'logsig'
    finv = 'ilogsig'
    fderiv = 'dlogsig'
        
    local_matrices = []
    
    process_pool = Pool(nodes=procesos)
    
    # #############################################################################
    # ENTRENAMIENTO DEL ENCODER 
    num_capas = len(arquitectura)
    if num_capas >= 3 and (arquitectura[0] == arquitectura[len(arquitectura)-1]):
        capa_intermedia = np.argmin(arquitectura)
        m = arquitectura[0]
        Hs_encoder_list = []
        Ws_encoder_list = []
        
        # Primera capa entrenada con DSVD
        Xs = split_optim(X,procesos)
        H1, W1, Up, Sp = calcularPesos1CapaEncoder (Xs, arquitectura[1], fh, procesos, process_pool)
        Hs_encoder_list.append(H1)
        Ws_encoder_list.append(W1)
        local_matrices.append((Up, Sp))

        # Siguientes capas del encoder
        if capa_intermedia > 1:
            for i in range (2, capa_intermedia+1):
                H1 = split_optim(Hs_encoder_list[-1],procesos)
                H2, W2, Up, Sp = calcularPesos1CapaEncoder (H1, arquitectura[i], fh, procesos, process_pool)
                Hs_encoder_list.append(H2)
                Ws_encoder_list.append(W2)
                local_matrices.append((Up, Sp))

        # #############################################################################
        # ENTRENAMIENTO DEL DECODER EXCEPTO LA ÚLTIMA CAPA 
        Hs_decoder_list = []
        Ws_decoder_list = []
        count = 0
        if capa_intermedia != num_capas-2:    
            for j in range (capa_intermedia+1, num_capas-1):
                if j == capa_intermedia+1:
                    H4, W4, M4, U4, S4 = calcularPesos1CapaDecoder(Hs_encoder_list[-1].T, fh, finv, fderiv, lam_HL, process_pool, W_list[count], np.repeat(B_list[count], Hs_encoder_list[-1].T.shape[0], axis = 0))
                    for q in range (0, len(S4)):
                        S4[q] = S4[q].toarray().diagonal()
                    H4 = H4[1:, :]
                    Hs_decoder_list.append(H4)
                    Ws_decoder_list.append(W4)
                    local_matrices.append((M4, U4, S4))
                elif j > capa_intermedia+1:
                    H4, W4, M4, U4, S4 = calcularPesos1CapaDecoder(Hs_decoder_list[-1].T, fh, finv, fderiv, lam_HL, process_pool, W_list[count], np.repeat(B_list[count], Hs_encoder_list[-1].T.shape[0], axis = 0))
                    for q in range (0, len(S4)):
                        S4[q] = S4[q].toarray().diagonal()
                    H4 = H4[1:, :]
                    Hs_decoder_list.append(H4)
                    Ws_decoder_list.append(W4)
                    local_matrices.append((M4, U4, S4))
                count = count + 1
        
            # #############################################################################    
            # CÁLCULO DE LA ÚLTIMA CAPA
            W6 = []; H6 = []; M6_list = []; U6_list = []; S6_list = []
            for i in range (0, m):
                args = onelayer_reg((Hs_decoder_list[-1], X[i, :], 'ilinear', 'dlinear', lam_LL, None, None, None))
                W6_aux, M6, U6, S6 = args
                H6_aux = nnsimul(W6_aux, Hs_decoder_list[-1], fn)
                W6.append(W6_aux[1:])
                H6.append(H6_aux)
                M6_list.append(M6)
                U6_list.append(U6)
                S6 = S6.toarray().diagonal()
                S6_list.append(S6)
            W6 = np.vstack(W6).T
            H6 = np.vstack(H6)
            Hs_decoder_list.append(H6)
            Ws_decoder_list.append(W6)
            local_matrices.append((M6_list, U6_list, S6_list))
        else:
        
            # #############################################################################    
            # CÁLCULO DE LA ÚLTIMA CAPA
            W6 = []; H6 = []; M6_list = []; U6_list = []; S6_list = []
            for i in range (0, m):
                args = onelayer_reg((Hs_encoder_list[-1], X[i, :], 'ilinear', 'dlinear', lam_LL, None, None, None))
                W6_aux, M6, U6, S6 = args
                H6_aux = nnsimul(W6_aux, Hs_encoder_list[-1], fn)
                W6.append(W6_aux[1:])
                H6.append(H6_aux)
                M6_list.append(M6)
                U6_list.append(U6)
                S6 = S6.toarray().diagonal()
                S6_list.append(S6)
            W6 = np.vstack(W6).T
            H6 = np.vstack(H6)
            Hs_decoder_list.append(H6)
            Ws_decoder_list.append(W6)
            local_matrices.append((M6_list, U6_list, S6_list))
    else:
        print(">>> ERROR: Arquitectura inapropiada: ", arquitectura)
    net_parameters = Ws_encoder_list, Ws_decoder_list, fh, fn  
    
    return Hs_decoder_list[-1], net_parameters, local_matrices          

def DAEF_train_incrementally(local_matrices, external_matrices, arquitectura, lam_HL, lam_LL, procesos):
    # Entrenamiento local 
    fn = 'linear'
    fh = 'logsig'
    finv = 'ilogsig'
    fderiv = 'dlogsig'
        
    final_matrices = []
    
    process_pool = Pool(nodes=procesos)
    
    # #############################################################################
    # ENTRENAMIENTO DEL ENCODER 
    num_capas = len(arquitectura)
    if num_capas >= 3 and (arquitectura[0] == arquitectura[len(arquitectura)-1]):
        capa_intermedia = np.argmin(arquitectura)
        m = arquitectura[0]
        Ws_encoder_list = []
        
        # Primera capa entrenada con DSVD
        W1, Up, Sp = calcularPesos1CapaEncoderIncremental(arquitectura[1], local_matrices[0], external_matrices[0])
        Ws_encoder_list.append(W1)
        final_matrices.append((Up, Sp))

        # Siguientes capas del encoder
        if capa_intermedia > 1:
            for i in range (2, capa_intermedia+1):
                W2, Up, Sp = calcularPesos1CapaEncoderIncremental(arquitectura[i], local_matrices[i-1], external_matrices[i-1])
                Ws_encoder_list.append(W2)
                final_matrices.append((Up, Sp))
        # #############################################################################
        # ENTRENAMIENTO DEL DECODER EXCEPTO LA ÚLTIMA CAPA 
        Ws_decoder_list = []
        count = 0
        if capa_intermedia != num_capas-2:    
            for j in range (capa_intermedia+1, num_capas-1):
                if j == capa_intermedia+1:
                    W4, M4, U4, S4 = calcularPesos1CapaDecoderIncremental(local_matrices[j-1], external_matrices[j-1], lam_HL, process_pool)
                    for q in range (0, len(S4)):
                        S4[q] = S4[q].toarray().diagonal()
                    Ws_decoder_list.append(W4)
                    final_matrices.append((M4, U4, S4))
                elif j > capa_intermedia+1:
                    W4, M4, U4, S4 = calcularPesos1CapaDecoderIncremental(local_matrices[j-1], external_matrices[j-1], lam_HL, process_pool)
                    for q in range (0, len(S4)):
                        S4[q] = S4[q].toarray().diagonal()
                    Ws_decoder_list.append(W4)
                    final_matrices.append((M4, U4, S4))
                count = count + 1

            # #############################################################################    
            # CÁLCULO DE LA ÚLTIMA CAPA
            M6_list = []; U6_list = []; S6_list = []
            MUS_antes = external_matrices[-1][0], external_matrices[-1][1], external_matrices[-1][2]
            MUS_nueva = local_matrices[-1][0], local_matrices[-1][1], local_matrices[-1][2]
            args = calcularPesos1CapaDecoderIncremental(MUS_antes, MUS_nueva, lam_LL, process_pool)
            W6_aux, M6, U6, S6 = args
            M6_list.append(M6)
            U6_list.append(U6)
            for q in range (0, len(S6)):
                S6[q] = S6[q].toarray().diagonal()
            S6_list.append(S6)
            Ws_decoder_list.append(W6_aux.T[1:])
            final_matrices.append((M6_list, U6_list, S6_list))
        else:
        
            # #############################################################################    
            # CÁLCULO DE LA ÚLTIMA CAPA
            M6_list = []; U6_list = []; S6_list = []
            MUS_antes = external_matrices[-1][0], external_matrices[-1][1], external_matrices[-1][2]
            MUS_nueva = local_matrices[-1][0], local_matrices[-1][1], local_matrices[-1][2]
            args = calcularPesos1CapaDecoderIncremental(MUS_antes, MUS_nueva, lam_LL, process_pool)
            W6_aux, M6, U6, S6 = args
            M6_list.append(M6)
            U6_list.append(U6)
            for q in range (0, len(S6)):
                S6[q] = S6[q].toarray().diagonal()
            S6_list.append(S6)
            Ws_decoder_list.append(W6_aux.T[1:])
            final_matrices.append((M6_list, U6_list, S6_list))
    else:
        print(">>> ERROR: Arquitectura inapropiada: ", arquitectura)
    net_parameters = Ws_encoder_list, Ws_decoder_list, fh, fn  
    
    return net_parameters, final_matrices      

def DAEF_predict(X, net_parameters):
      
    # #############################################################################    
    # CÁLCULO CONJUNTO DE TEST
    Ws_encoder_list, Ws_decoder_list, fh, fn = net_parameters
    
    for i in range (0, len(Ws_encoder_list)):
        X = eval(fh)(Ws_encoder_list[i].T @ X)
    for j in range (0, len(Ws_decoder_list)):
        if j == (len(Ws_decoder_list)-1):
            reconstruction = eval(fn)(Ws_decoder_list[j].T @ X)

        else:
            X = eval(fh)(Ws_decoder_list[j].T @ X)
                
        if j < len(Ws_decoder_list)-1:
            # Eliminamos la fila que contiene el bias a la salida de cada capa del decoder (excepto en la salida final)
            X = X[1:, :]
            if j == (len(Ws_decoder_list)-2):
                0
    return reconstruction                    

def compute_train_error (X, prediction, umbral_tipo):
    
    erroresTrain = ((prediction - X)**2).T
    erroresTrain = np.mean(erroresTrain,1)
    
    if umbral_tipo == "atipico":
        q3_train, q1_train = np.percentile(erroresTrain, [75 ,25])
        iqr_train = q3_train - q1_train
        threshold = q3_train + 1.5 * iqr_train
    elif umbral_tipo == "extremadamente atipico":
        q3_train, q1_train = np.percentile(erroresTrain, [75 ,25])
        iqr_train = q3_train - q1_train
        threshold = q3_train + 3 * iqr_train
    else:
        threshold = np.percentile(erroresTrain, [umbral_tipo])[0]
    
    return threshold
    
def predict_and_clasify (X, Y, net_parameters, threshold):
    
    prediction = DAEF_predict(X, net_parameters)
    erroresTest = ((prediction - X)**2).T
    erroresTest = np.mean(erroresTest,1)
    clasification = np.where(erroresTest < threshold, 0, 1)
    cm = calcular_metricas(Y, clasification, "")
    return clasification
    
def train_normalizer (X):
    scaler = preprocessing.StandardScaler().fit(X) 
    return scaler

def normalize_data (X, scaler):
    X_normalized = pd.DataFrame(scaler.transform(X))
    return X_normalized