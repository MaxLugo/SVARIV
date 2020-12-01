import numpy as np
import matplotlib.pyplot as plt

def ols(Y, X):
    '''
    Input:
        Y = array or matrix endogenous t as rows and n as columns (n=endog variables)  
        X = array or matrix exogenous in the regressions
    Output:
        rv = dictionary with betas_hat, errors matrix and Y estimated
    '''
    betas = np.linalg.inv(X.T @ X) @ X.T @ Y
    Y_hat = X @ betas
    errores = Y - Y_hat
    rv = {'betas_hat':betas, 'errors':errores, 'Y_hat':Y_hat}
    return rv

def first_stage_f(y, Z, kz=None):
    '''
    Input:
        y = vector with the endogenous variable to instrument (type np array)
        Z = vector/matrix of instruments (type np array)
        kz = integer. If Z have control variables kz is the first column of Z
        that have control variable. If kz = none (default value) all variables
        will be consider has instruments
    Output:
        rv = dictionary with 3 F statistics: non robust, effective and robust
    Note: it assumes that Z has to be order, first columns as instruments and 
    the following are controls (if it is the case). 
    '''
    #FIXME This is not correct! Do not use this function!
    z = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
    Pi = np.linalg.inv(z.T @ z) @ z.T @ y
    if  kz != None:
        z, Pi = z[:,:kz], Pi[:kz,:]
    e = y - z @ Pi
    Pz = z @ np.linalg.inv(z.T @ z) @ z.T
    o = (e.T @ e / len(e))[0,0]
    F_nonrobust = y.T @ Pz @ y / (len(z.T)*o)
    rv = {'F_nonrobust':F_nonrobust[0,0]}
    
    #effective F
    O = np.kron(z, e[:,0]) / len(y)**.5
    J = int(len(O.T) / len(O))
    v = np.array([np.diag(O[:, j*len(O):(j+1)*len(O)]) for j in range(J)]).T
    W = v.T @ v
    F_effective = (y.T @ Pz @ y) / np.trace(W)
    rv['F_effective'] = F_effective[0,0]
    # equivalent to :
    # O = np.array([list(z[:,j]*e[:,0]/len(y)**.5) for j in range(len(z.T))]).T
        
    #robust F
    F_robust = (y.T @ z @ np.linalg.inv(W) @ z.T @ y) / (len(z)*len(z.T))
    rv['F_robust'] = F_robust[0,0]
    return rv


def NW_hac_STATA(Vars, lags):
    '''
    Input: 
        Vars = variables matrix (has to be ordered in the same way of the model)
        lags = number of lags (scalar) to be use in the algorithm
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
        X = matrix of exogenous variables for ols estimation (np.array) 
        Z = matrix or vector of exog instruments (np.array)
        eta = matrix of errors (np.array)
        p = # of lags (scalar)
        n = # of endog (scalar)
        nvar = number of the endog variable to be use in the test
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


def irf_lineal_cholesky(betas, S, periods=21, normalize=True, cumulative=True):
    '''
    Input:
        betas = array in the order of the model (np.array) no constant 
                with n rows = number of endog variables, in the paper betas=A
        S = is a covariance matrix (np.array)
            if S has shape nx1 it will asume that S = Gamma_hat
        normalize=True will normalize to the shock to be 1 default=True
        cumulative=True will accumulate the irf response default=True
        periods = scalar with the number of periods to use in the irf default=21
    Output:
        irf = matrix of irf (np.array) with all irf for all endog variables
    Notice the shock is always over the first variable. 
    '''
    use_chol = True if S.shape[1] >= 2 else False
    Lags = int(betas.shape[1] / betas.shape[0]) 
    num_endog = int(betas.shape[0]) 
    betas_lags = [betas[:, i*num_endog:(i+1)*num_endog] for i in range(Lags)]
    irf = np.zeros((periods, betas.shape[0]))
    irf_0 = list(np.linalg.cholesky(S)[:,0]) if use_chol==True else list(S)
    if normalize:
        irf_0 = np.array(irf_0) / np.array(irf_0)[0]
    irf[0,:] = list(irf_0)
    for t in range(1, periods):
        B = betas_lags[:t]
        i_lags = irf[0:t]
        i_lags = i_lags[-len(B):,:]
        i_lags_list = [i_lags[j,:].reshape(num_endog, 1) for j in range(i_lags.shape[0])]
        i_lags_list.reverse()
        agregg = [z[0] @ z[1] for z in zip(B, i_lags_list)]
        sum_t = np.concatenate(agregg, axis=1).sum(axis=1)
        irf[t,:] = sum_t
    if cumulative:
        irf_cum = irf.cumsum(axis=0)
    rv = irf_cum if cumulative else irf
    return rv


def MA_representation(betas, p, hori=21):    
    '''
    Input:
        betas = matrix with the betas of the reduce ols estimation with
                n rows = number endog variables and columns = number of p lags * n
                in the paper betas=A.
        p = number of lags
        hori = number of forecast periods to be use in the MA representation 
                in the paper is 20. In python is
                going to be 21 because it is not inclusive
    Output:
        C = list of matrixes (0 to k elements) with the MA representation 
            in the sense of
            Y_t = Sum_k=0^inf C_k(A) * e_t-k
            where:
                C(A)_k = Sum_m=1^k C_k-m(A) A_m for k=1,2,...    
    Notes: betas_lag doesn't consider the constant of course. 
            For values m>p => A_m=0.
    '''
    n = len(betas)
    A, C = [betas[:, n*i: n*(i+1)]  for i in range(p)], [np.eye(n)]
    for m in range(1, hori):
        C_m = np.array([a @ c for a, c in zip(A[:m], list(reversed(C)))]).sum(axis=0)
        C.append(C_m)
    return C


def Gmatrices(betas, p, hori=21):
    '''
    Input:
        betas = parameters of the ols reduce estimation (no constant).
                n rows = number endog variables and columns = number of p lags * n
                in the paper betas=A.
        p = number of lags in the var
        horin = number of forecast periods
    Output:
        rv = {'G': gradient matrix for a given horizon,
              'Gcum':G (cumulative)}
    Notes: it is 3D array elements in the dictionary rv. The 2 axis is the horizon
    axis
    '''
    n = len(betas)
    
    #create the MA representation
    C = MA_representation(betas, p, 21)
    C_aux = ([np.eye(n)] + C[:hori])[:hori-1]
    
    J = np.concatenate([np.eye(n), np.zeros((n,(p-1)*n))], axis=1)
    
    Alut = np.concatenate([betas, np.zeros((len(betas.T)-len(betas), len(betas.T)))], axis=0)
    Alut[len(betas):,:len(Alut.T)-len(betas)] = np.eye(len(Alut[len(betas):,:]))
    
    AJ = [np.linalg.matrix_power(Alut, h) @ J.T for h in range(hori-1)]
    AJp = np.concatenate(AJ, axis=1).T
    
    AJaux = []
    for i in range(hori-1):
        l = np.kron(AJp[:n*(hori- i),:], C_aux[i])
        Z = np.zeros((AJp.shape[0]*n, AJp.shape[1]*n))
        Z[i*n**2:, :] = l[:len(Z)-n**2*i, :]
        AJaux.append(Z)
    
    G0 = np.array(AJaux).sum(axis=0).T.reshape(p*n**2, n**2, hori-1, order='F')
    Gaux = np.moveaxis(G0, [0,1,2], [1,0,2])
    G = np.zeros((Gaux.shape[0], Gaux.shape[1], Gaux.shape[2]+1))
    
    for i in range(1,Gaux.shape[2]+1):
        G[:,:,i] = Gaux[:,:,i-1]
    
    Gcum = np.cumsum(G,2)
    
    return {'G':G, 'Gcum':Gcum}



def simple_plot(title, est, est2, low, upp, xticks, ylabel, rot=0, ylim0=-3.5, 
                ylim1=3.5, figsize=(20,5), xlabel='Months'):
    '''
    Input:
        title = title of the plot
        est = Gamma estimation (list of np array)
        est2 = cholesky esitmaiton (list of np array)
        low = lower interval (list of np array)
        upp = upper interval (list of np array)
        xticks = list of np array with the xticks of the plot
        ylabel = Y title
        rot = rotation of the xticks (scalar) default=0
        ylim0 = lower limit of the yaxis (scalar) default=-3.5
        ylim1 = upper limit of the yaxis (scalar) default=3.5
        figsize = (x,y) x=size of xaxis and y= size of yaxis default=(20,5)
        xlabel = title of xaxis default=Months
    Output:
        figure of matplotlib
    '''
    f, ax = plt.subplots(1, figsize=figsize)
    ax.set_title(title,fontsize=30)
    plt.plot(est, color='blue', zorder=1, linewidth=4)
    plt.plot(est2, color='red', zorder=1, linewidth=1)
    plt.fill_between(xticks , low, upp, color='b', alpha=.1)
    ax.legend(['SVARIV','Cholesky','$CS^{boots}$'], fontsize=17).set_zorder(5)
    plt.xticks(list(range(len(xticks))), xticks, rotation=rot,fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(ylim0, ylim1)
    plt.xlim(0, 20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.tight_layout()
    plt.show()
    return f



