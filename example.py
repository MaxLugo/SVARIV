import pandas as pd
import numpy as np
import SVARIV
#from scipy.stats import chi2



#parameters 
p = 24 # lags
n = 3 # endog variables
nvar = 1 # test over the first one


#=============================================================================#
#
# Load the data
#
#=============================================================================#
X = pd.read_excel('./data/Oil/Data.xls', header=None)
X = np.concatenate([np.array([X[j].shift(i) for j in range(n)]).T for i in range(1, p+1)], axis=1)
X = X[p:, :] 
X = np.concatenate([np.ones((len(X),1)), X], axis=1)
Z = pd.read_excel('./data/Oil/ExtIV.xls', header=None).values[p:,:]
Y = pd.read_excel('./data/Oil/Data.xls', header=None).values[p:,:]


#=============================================================================#
#
# ols estimation
#
#=============================================================================#
ols_est = SVARIV.ols(Y, X)
eta = ols_est['errors']


#=============================================================================#
#
# Gamma_hat & Wald
#
#=============================================================================#
WHat, wald, Gamma_hat = SVARIV.get_gamma_wald(X, Z, eta, p, n, nvar)
print('Gamma_hat estimated: {}'.format(Gamma_hat))
print('Wald: {}'.format(wald))


#=============================================================================#
#
# Impulse response in construction
#
#=============================================================================#
betas = ols_est['betas_hat'].T # A with constant
betas_lag = betas[:,1:] # A without constant
omega = (ols_est['errors'].T @ ols_est['errors']) / len(ols_est['errors']) #covar matrix errors

#Puntual estimation
# irf with cholesky decomposition
irf_chol = SVARIV.irf_lineal_cholesky(betas_lag, omega, cumulative=False)
irf_chol_cum = SVARIV.irf_lineal_cholesky(betas_lag, omega, cumulative=True)

# irf with Gamma_hat
irf_gamma = SVARIV.irf_lineal_cholesky(betas_lag, Gamma_hat, cumulative=False)
irf_gamma_cum = SVARIV.irf_lineal_cholesky(betas_lag, Gamma_hat, cumulative=True)


#=============================================================================#
#
# 68% confidence set
#
#=============================================================================#
# Confidense sets dmethod
C = SVARIV.MA_representation(betas_lag, p, hori=21)  # MA representation
Ccum = [np.array(C[:i+1]).sum(axis=0) for i in range(len(C))]
G = SVARIV.Gmatrices(betas_lag, p, hori=21)['G'] # Gradient matrix
Gcum = SVARIV.Gmatrices(betas_lag, p, hori=21)['Gcum'] # Gradient matrix cumulative
T = len(Y) # number of obs
CI = SVARIV.CI_dmethod(Gamma_hat, WHat, G, T, C, hori=21, confidence=0.68, scale=1, nvar=1) #CI intervals
CI_cum = SVARIV.CI_dmethod(Gamma_hat, WHat, Gcum, T, Ccum, hori=21, confidence=0.68, scale=1, nvar=1) #CI intervals
CI_s = SVARIV.CI_dmethod_standard(Gamma_hat, WHat, G, T, C)
CI_s_cum = SVARIV.CI_dmethod_standard(Gamma_hat, WHat, Gcum, T, Ccum)


# plugin CS figure 1.1 
plugin_cs = np.array([irf_gamma_cum[:,0] + SVARIV.norm_critval(confidence=0.68)*CI_s_cum['pluginirfstderror'][0,:], 
                      irf_gamma_cum[:,0] - SVARIV.norm_critval(confidence=0.68)*CI_s_cum['pluginirfstderror'][0,:]]).T
fig11 = SVARIV.simple_plot('Cumulative Percent in Global Oil Production (68% interval)', 
                          irf_gamma_cum[:,0], irf_chol_cum[:,0], 
                          CI_cum['l'][0,:], CI_cum['u'][0,:],
                          plugin_cs[:,0], plugin_cs[:,1],
                          list(range(len(irf_gamma_cum))), 
                          '', rot=0, ylim0=0, 
                          ylim1=1, figsize=(20,5))

# figure 1.2
plugin_cs = np.array([irf_gamma[:,1] + SVARIV.norm_critval(confidence=0.68)*CI_s['pluginirfstderror'][1,:], 
                      irf_gamma[:,1] - SVARIV.norm_critval(confidence=0.68)*CI_s['pluginirfstderror'][1,:]]).T
fig12 = SVARIV.simple_plot('Index of Real Economic Activity (68% interval)', 
                          irf_gamma[:,1], irf_chol[:,1],
                          CI['l'][1,:], CI['u'][1,:],
                          plugin_cs[:,0], plugin_cs[:,1],
                          list(range(len(irf_gamma_cum))), 
                          '', rot=0, ylim0=-.2, 
                          ylim1=.2, figsize=(20,5))

# figure 1.3
plugin_cs = np.array([irf_gamma[:,2] + SVARIV.norm_critval(confidence=0.68)*CI_s['pluginirfstderror'][2,:], 
                      irf_gamma[:,2] - SVARIV.norm_critval(confidence=0.68)*CI_s['pluginirfstderror'][2,:]]).T
fig13 = SVARIV.simple_plot('Real Price Oil (68% interval)', 
                          irf_gamma[:,2], irf_chol[:,2],
                          CI['l'][2,:], CI['u'][2,:],
                          plugin_cs[:,0], plugin_cs[:,1],
                          list(range(len(irf_gamma_cum))), 
                          '', rot=0, ylim0=-.4, 
                          ylim1=.2, figsize=(20,5))


#=============================================================================#
#
# 95% confidence set
#
#=============================================================================#
# Confidense sets dmethod
C = SVARIV.MA_representation(betas_lag, p, hori=21)  # MA representation
Ccum = [np.array(C[:i+1]).sum(axis=0) for i in range(len(C))]
G = SVARIV.Gmatrices(betas_lag, p, hori=21)['G'] # Gradient matrix
Gcum = SVARIV.Gmatrices(betas_lag, p, hori=21)['Gcum'] # Gradient matrix cumulative
T = len(Y) # number of obs
CI = SVARIV.CI_dmethod(Gamma_hat, WHat, G, T, C, hori=21, confidence=0.95, scale=1, nvar=1) #CI intervals
CI_cum = SVARIV.CI_dmethod(Gamma_hat, WHat, Gcum, T, Ccum, hori=21, confidence=0.95, scale=1, nvar=1) #CI intervals
CI_s = SVARIV.CI_dmethod_standard(Gamma_hat, WHat, G, T, C)
CI_s_cum = SVARIV.CI_dmethod_standard(Gamma_hat, WHat, Gcum, T, Ccum)


# plugin CS figure 1.1 
plugin_cs = np.array([irf_gamma_cum[:,0] + SVARIV.norm_critval()*CI_s_cum['pluginirfstderror'][0,:], 
                      irf_gamma_cum[:,0] - SVARIV.norm_critval()*CI_s_cum['pluginirfstderror'][0,:]]).T
fig11 = SVARIV.simple_plot('Cumulative Percent in Global Oil Production (95% interval)', 
                          irf_gamma_cum[:,0], irf_chol_cum[:,0], 
                          CI_cum['l'][0,:], CI_cum['u'][0,:],
                          plugin_cs[:,0], plugin_cs[:,1],
                          list(range(len(irf_gamma_cum))), 
                          '', rot=0, ylim0=-2, 
                          ylim1=2, figsize=(20,5))

# figure 1.2
plugin_cs = np.array([irf_gamma[:,1] + SVARIV.norm_critval()*CI_s['pluginirfstderror'][1,:], 
                      irf_gamma[:,1] - SVARIV.norm_critval()*CI_s['pluginirfstderror'][1,:]]).T
fig12 = SVARIV.simple_plot('Index of Real Economic Activity (95% interval)', 
                          irf_gamma[:,1], irf_chol[:,1],
                          CI['l'][1,:], CI['u'][1,:],
                          plugin_cs[:,0], plugin_cs[:,1],
                          list(range(len(irf_gamma_cum))), 
                          '', rot=0, ylim0=-1, 
                          ylim1=2, figsize=(20,5))

# figure 1.3
plugin_cs = np.array([irf_gamma[:,2] + SVARIV.norm_critval()*CI_s['pluginirfstderror'][2,:], 
                      irf_gamma[:,2] - SVARIV.norm_critval()*CI_s['pluginirfstderror'][2,:]]).T
fig13 = SVARIV.simple_plot('Real Price Oil (95% interval)', 
                          irf_gamma[:,2], irf_chol[:,2],
                          CI['l'][2,:], CI['u'][2,:],
                          plugin_cs[:,0], plugin_cs[:,1],
                          list(range(len(irf_gamma_cum))), 
                          '', rot=0, ylim0=-1, 
                          ylim1=3, figsize=(20,5))




