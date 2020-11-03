import pandas as pd
import SVARIV


#this file will import the data from a excel with the original data of the paper.
X = pd.read_excel('data_model.xlsx', sheet_name='X', header=None).values
Z = pd.read_excel('data_model.xlsx', sheet_name='Z', header=None).values
eta = pd.read_excel('data_model.xlsx', sheet_name='errors', header=None).values


#this is the gamma of the paper. 
Gamma = pd.read_excel('data_model.xlsx', sheet_name='Gamma', header=None).values
p = 24 #lags
n = 3 #endog variables
nvar = 1 #test over the first one


#this is the test of Wald (replication)
WHat, wald, Gamma_hat = SVARIV.get_gamma_wald(X, Z, eta, p, n, nvar)
print('difference between Gammas: ')
print(abs(Gamma - Gamma_hat))







