import pandas as pd
import numpy as np
from tqdm import tqdm
import SVARIV



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
print('Gamma_hat estimated: {}'.format(Gamma_hat))
print('Wald: {}'.format(wald))


#=============================================================================#
# Impulse response
#=============================================================================#
betas = ols_est['betas_hat'].T
betas_lag = betas[:,1:]
omega = (ols_est['errors'].T @ ols_est['errors']) / len(ols_est['errors'])

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












