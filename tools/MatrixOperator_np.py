import numpy as np

def Cal_Stat(U,priodic_dim): #arrange priodic_dim
    U_avg = U
    Reduced_dim = 0
    for i in range(len(priodic_dim)):
        U_avg = np.average(U_avg,axis=priodic_dim[i]-Reduced_dim)
        Reduced_dim +=1
    return U_avg

def TransferMatrix(Matrix,steps): # Move the sliding window forward by steps.
    if steps >= 0:
        New_Matrix = np.concatenate((Matrix[steps:],Matrix[:steps]))
    else:
        steps = np.abs(steps)
        New_Matrix = np.concatenate((Matrix[-steps:],Matrix[:-steps]))
            
    return New_Matrix


def TransferMatrix_alongAxis(Matrix,axis,steps):
    return np.apply_along_axis(TransferMatrix,axis,Matrix,steps)
