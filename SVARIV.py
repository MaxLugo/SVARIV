import numpy as np


def NW_hac_STATA(Vars, lags):
    '''
    Input: 
        Vars = matrix variables has to be ordered
        lags = number of lags
    Output:
        corrected sigma by lags
    '''
    Sigma0 = (Vars.T @ Vars) / len(Vars)
    sigma_cov_k = lambda V,k: (V[0:-k].T @ V[0:-k]) / len(V)
    S = Sigma0
    for n in range(lags):
        S += (1 - n/(lags+1)) * sigma_cov_k(Vars, n) @ sigma_cov_k(Vars, n).T
    return S


def get_gamma_wald(X, Z, eta, p, n, nvar):
    '''
    Input:
        X = matrix of exogenous variables for ols estimation type = np.array 
        Z = matrix or vector of exog instruments type = np.array
        eta = matrix of errors type = np.array
        p = # of lags (scalar)
        n = # of endog (scalar)
        nvar = number to  endog variables to be use in the test
    output:
        WHat, wald, Gamma_hat
    '''
    Gamma_hat = ((Z.T @ eta) / len(eta)).T
    matagg = np.concatenate((X, eta, Z), axis=1).T
    auxeta = np.zeros((len(eta.T), len(matagg), len(matagg.T)))
    val = np.array([eta[i, np.newaxis].T @ matagg[:,i,np.newaxis].T for i in range(len(matagg.T))])
    val_mean = val.mean(axis=0)
    val_minus_mean = val - val_mean
    for i in range(auxeta.shape[2]):
        auxeta[:,:,i] = val_minus_mean[i,:,:]
    
    AuxHAC2 = [auxeta[:,:,i].reshape(1,np.multiply(*auxeta[:,:,0].shape), order='F') 
               for i in range(auxeta.shape[2])]
    AuxHAC2 = np.concatenate(AuxHAC2, axis=0)
    AuxHAC3 = NW_hac_STATA(AuxHAC2, 0)
    
    I = np.eye(len(eta.T))
    V = np.kron(I[0,:], I)
    for i in range(1, len(eta.T)):
        V = np.concatenate((V, np.kron(I[i,:], I[i:,:])), axis=0)
    
    Q1 = X.T @ X / len(X)
    Q2 =  Z.T @ X / len(X) 
    m = X.shape[1] - (n*p)
    
    Shat = np.zeros((len(Q1) * eta.shape[1] + len(V), len(Q1) * eta.shape[1] + len(V.T) + n))
    A = np.kron(np.concatenate((np.zeros((n*p, m)), np.eye(n*p)), axis=1) @ np.linalg.inv(Q1),
                np.eye(n))
    Shat[0:A.shape[0], 0:A.shape[1]] = A
    Shat[A.shape[0]:A.shape[0]+V.shape[0], A.shape[1]:A.shape[1]+V.shape[1]] = V
    B = -np.kron(Q2 @ np.linalg.inv(Q1), np.eye(n))
    Shat[-B.shape[0]:,0:B.shape[1]] = B
    C = np.eye(B.shape[0])
    Shat[-C.shape[0]:,-C.shape[1]:] = C
    WHataux  = Shat @ AuxHAC3 @ Shat.T
        
    L = len(Q1) * len(C)
    WHat = np.zeros((L, L))
    WHat[:-len(C), :-len(C)] = WHataux[:L-len(C), :L-len(C)]
    WHat[-len(C):, -len(C):] = WHataux[-len(C):, -len(C):]
    WHat[-len(C):, :L-len(C)] = WHataux[-len(C):, :L-len(C)]
    WHat[:-len(C), L-len(C):] = WHat[-len(C):, :L-len(C)].T
        
    place = -len(C) + nvar - 1
    wald = (len(eta)) * Gamma_hat[nvar-1, 0]**2 / WHat[place, place]
    return WHat, wald, Gamma_hat








