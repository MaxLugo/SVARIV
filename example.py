import pandas as pd
import SVARIV

#=============================================================================#
# Load the data
#=============================================================================#
X = pd.read_excel('./data/data_model.xlsx', sheet_name='X', header=None).values
Z = pd.read_excel('./data/data_model.xlsx', sheet_name='Z', header=None).values
eta = pd.read_excel('./data/data_model.xlsx', sheet_name='errors', header=None).values



#parameters 
p = 24 #lags
n = 3 #endog variables
nvar = 1 #test over the first one


#this is the test of Wald (replication)
WHat, wald, Gamma_hat = SVARIV.get_gamma_wald(X, Z, eta, p, n, nvar)
print('Gamma_hat estimated: {}'.format(Gamma_hat))







