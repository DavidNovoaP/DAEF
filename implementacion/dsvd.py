import numpy as np
import math
import multiprocessing as mp
import time
import sys
from pathos.multiprocessing import ProcessingPool as Pool

def LocalSVD(mat): # Calculo de la SVD local
    import numpy as np
    U, S, V = np.linalg.svd(mat, full_matrices = False); # economy SVD
    return U@np.diag(S)


def dsvd(LocalData, n, procs, process_pool):
    import numpy as np
    result = list(process_pool.imap(LocalSVD, LocalData))
    
    Up, Sp, Vp = np.linalg.svd(np.concatenate(result, axis=1), full_matrices = False); # economy
    Sp = np.diag(Sp);
    return Up, Sp;
    
