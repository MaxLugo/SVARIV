import pandas as pd
import numpy as np
from tqdm import tqdm
import SVARIV
from scipy.stats import chi2
from scipy.stats import norm


#parameters 
p = 24 #lags
n = 3 #endog variables
nvar = 1 #test over the first one


#=============================================================================#
# Load the data
#=============================================================================#
X = pd.read_excel('./data/Oil/Data.xls', header=None)
X = np.concatenate([np.array([X[j].shift(i) for j in range(n)]).T for i in range(1, p+1)], axis=1)
X = X[p:, :] 
X = np.concatenate([np.ones((len(X),1)), X], axis=1)
Z = pd.read_excel('./data/Oil/ExtIV.xls', header=None).values[p:,:]
Y = pd.read_excel('./data/Oil/Data.xls', header=None).values[p:,:]


#=============================================================================#
# ols estimation
#=============================================================================#
ols_est = SVARIV.ols(Y, X)
eta = ols_est['errors']


#=============================================================================#
# Gamma & Wald
#=============================================================================#
WHat, wald, Gamma_hat = SVARIV.get_gamma_wald(X, Z, eta, p, n, nvar)
F_stats = SVARIV.first_stage_f(Y[:,0].reshape(len(Y),1), 
                               np.concatenate([Z, X[:,1:]], axis=1), kz=1)
F_stats = SVARIV.first_stage_f(Y[:,0].reshape(len(Y),1), Z)



print('Gamma_hat estimated: {}'.format(Gamma_hat))
print('Wald: {}'.format(wald))
#print('First stage effective with controls F: {}'.format(F_stats['F_effective']))
#print('First stage nonrobust with controls F: {}'.format(F_stats['F_nonrobust']))
#print('critical value Chi2(1) at 95%: {}'.format(chi2.ppf(.95, 1)))




#=============================================================================#
# Impulse response in construction
#=============================================================================#
betas = ols_est['betas_hat'].T
betas_lag = betas[:,1:]
omega = (ols_est['errors'].T @ ols_est['errors']) / len(ols_est['errors'])
G = SVARIV.Gmatrices(betas_lag, p, hori=21)['G']
C = SVARIV.MA_representation(betas_lag, p, hori=21)
hori = 21
confidence = 0.95 #not in percent


#critical value
critval = norm.ppf(1-(1-confidence)/2)**2

#Get the W block matrixes
W1, W2 = WHat[:-n,:-n], WHat[-n:,-n:]
W12 = WHat[W1.shape[0]:,:W1.shape[0]].T

#initialize the values
e = np.eye(n)
ahat = np.zeros((n, hori))
bhat = np.zeros((n, hori))
chat = np.zeros((n, hori))
Deltahat = np.zeros((n, hori))
MSWlbound = np.zeros((n, hori))
MSWubound = np.zeros((n, hori)) 
casedummy = np.zeros((n, hori))

T = 356

scale=1


for j in range(n):    
    for ih in range(hori):
        ahat[j,ih] = T*(Gamma_hat[nvar-1, 0]**2) - critval*W2[nvar-1, nvar-1] 
        
        bhat[j, ih] = (-2*T*scale*(e[:,j].T @ C[ih] @ Gamma_hat @ Gamma_hat[nvar-1])
                       + 2*critval*scale*np.kron(Gamma_hat.T, e[:,j].T) @ G[:,:,ih] @ W12[:,nvar-1]
                       + 2*critval*scale*e[:,j].T @ C[ih] @ W2[:, nvar-1]
                       )
        
        chat[j, ih] = ( ((T**.5)*scale*e[:,j].T @ C[ih] @ Gamma_hat)**2
                       -critval*(scale**2)*np.kron(Gamma_hat.T, e[:,j].T) @ G[:,:,ih] @ W1 @ (np.kron(Gamma_hat.T, e[:,j].T) @ G[:,:,ih]).T
                       -2*critval*(scale**2)*np.kron(Gamma_hat.T, e[:,j].T) @ G[:,:,ih] @ W12 @ C[ih].T @ e[:,j]
                       -critval*(scale**2)*e[:,j].T @ C[ih] @ W2 @ C[ih].T @ e[:,j]
                      )
        
        Deltahat[j,ih] = bhat[j,ih]**2 - (4*ahat[j,ih] * chat[j,ih])
        
        if (ahat[j, ih]>0) and (Deltahat[j,ih]>0):            
            casedummy[j,ih] = 1
            MSWlbound[j,ih] = (-bhat[j,ih] - (Deltahat[j,ih]**.5))/(2*ahat[j,ih])
            MSWubound[j,ih] = (-bhat[j,ih] + (Deltahat[j,ih]**.5))/(2*ahat[j,ih])
        elif (ahat[j, ih]<0) and (Deltahat[j,ih]>0):
            casedummy[j,ih] = 2
            MSWlbound[j,ih] = (-bhat[j,ih] + (Deltahat[j,ih]**.5))/(2*ahat[j,ih])
            MSWubound[j,ih] = (-bhat[j,ih] - (Deltahat[j,ih]**.5))/(2*ahat[j,ih])
        elif (ahat[j, ih]>0) and (Deltahat[j,ih]<0):
            casedummy[j,ih] = 3
            MSWlbound[j,ih] = np.nan
            MSWubound[j,ih] = np.nan
        else:
            casedummy[j,ih] = 4
            MSWlbound[j,ih] = -np.inf
            MSWubound[j,ih] = np.inf
            
MSWlbound[nvar-1, 0] = scale
MSWubound[nvar-1, 0] = scale
        
    
















#Puntual estimation
#irf with cholesky decomposition
irf_chol = SVARIV.irf_lineal_cholesky(betas_lag, omega, cumulative=False)
irf_chol_cum = SVARIV.irf_lineal_cholesky(betas_lag, omega, cumulative=True)

#irf with Gamma_hat
irf_gamma = SVARIV.irf_lineal_cholesky(betas_lag, Gamma_hat, cumulative=False)
irf_gamma_cum = SVARIV.irf_lineal_cholesky(betas_lag, Gamma_hat, cumulative=True)


#create the confidence sets for the previous estimations
#I presented a usual bootstraps for th estimations not the presented in the paper
#I used the gamma_hat estimator for the bootstrap procedure
reps = 1000 #number of repetitions in the bootstrap
irf_gamma_b, irf_gamma_cum_b = np.zeros((21, n, reps)), np.zeros((21, n, reps))


#notice I did not fixed the seed in the random process (to be done)
for r in tqdm(range(reps)):
    random_pos = np.random.randint(0, len(Y), len(Y)) 
    ols_est_rand = SVARIV.ols(Y[random_pos], X[random_pos])
    eta_rand = ols_est_rand['errors']
    betas_rand = ols_est_rand['betas_hat'].T
    betas_lag_rand = betas_rand[:, 1:]
    omega_rand = (eta_rand.T @ eta_rand) / len(eta_rand)
    _, _, Gamma_hat_rand = SVARIV.get_gamma_wald(X[random_pos], Z[random_pos], eta_rand, p, n, nvar)
    irf_gamma_b[:, :, r] = SVARIV.irf_lineal_cholesky(betas_lag_rand, Gamma_hat_rand, cumulative=False)
    irf_gamma_cum_b[:, :, r] = SVARIV.irf_lineal_cholesky(betas_lag_rand, Gamma_hat_rand, cumulative=True)

conf = 68 #confidence set 


#plot Figure 1 (different confidence sets) Oil production
cs = np.sort(np.concatenate([irf_gamma_cum_b[:,:,i][:,0:1] for i in range(reps)], axis=1), axis=1)
cs = [[np.percentile(cs[i,:], (100-conf)/2) , np.percentile(cs[i,:], conf + (100-conf)/2)] for i in range(21)]
cs = np.array(cs)
fig11 = SVARIV.simple_plot('Cumulative Percent in Global Oil Production', 
                          irf_gamma_cum[:,0], irf_chol_cum[:,0], cs[:,0], cs[:,1],  
                          list(range(len(cs))), 
                          '', rot=0, ylim0=0, 
                          ylim1=1, figsize=(20,5))


#plot Figure 1 (different confidence sets) index of real economic activity
cs = np.sort(np.concatenate([irf_gamma_b[:,:,i][:,1:2] for i in range(reps)], axis=1), axis=1)
cs = [[np.percentile(cs[i,:], (100-conf)/2) , np.percentile(cs[i,:], conf + (100-conf)/2)] for i in range(21)]
cs = np.array(cs)
fig12 = SVARIV.simple_plot('Index of Real Economic Activity', 
                          irf_gamma[:,1], irf_chol[:,1], cs[:,0], cs[:,1],  
                          list(range(len(cs))), 
                          '', rot=0, ylim0=-0.2, 
                          ylim1=0.2, figsize=(20,5))


#plot Figure 1 (different confidence sets) Real Price of Oil
cs = np.sort(np.concatenate([irf_gamma_b[:,:,i][:,2:] for i in range(reps)], axis=1), axis=1)
cs = [[np.percentile(cs[i,:], (100-conf)/2) , np.percentile(cs[i,:], conf + (100-conf)/2)] for i in range(21)]
cs = np.array(cs)
fig13 = SVARIV.simple_plot('Real Price Oil', 
                          irf_gamma[:,2], irf_chol[:,2], cs[:,0], cs[:,1],  
                          list(range(len(cs))), 
                          '', rot=0, ylim0=-0.5, 
                          ylim1=0.2, figsize=(20,5))













