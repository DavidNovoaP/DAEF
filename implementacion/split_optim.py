# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 19:55:01 2020

@author: Beatriz
"""

import numpy as np
import math

#########################################################################################            
# Syntax:
# ------
# Ais = split_optim(A, maxblocks)
#
# Parameters of the function:
# --------------------------
# A : Matrix (m x n)
# maxblocks : Maximum number of blocks (lo habitual será utilizar el número de workers disponibles en el escenario distribuido)
#
# Returns:
# -------
# Cell que contiene los bloques de la matriz A, como mucho puede tener tamaño 'maxblocks' (Ais)


def split_optim(A, maxblocks):
    m,n = np.shape(A);
    block_size = math.ceil(n/maxblocks);
    work = n;
    count = 1;
    Ais = [None] * maxblocks; 
    
    while (work >= block_size):
         work = work - block_size;
         step = (count-1)*block_size;
         Ais[count-1] = A[:, step:(step)+block_size]; 
         count = count +1;
         
    if (work>0):
        step = (count-1) * block_size;  
        Ais[count-1] = A[:, step:];    
    return Ais;

        

   