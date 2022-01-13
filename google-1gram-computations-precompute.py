#!/usr/bin/env python3

print('Initializing...')
import pickle
import os
import numpy as np
import pandas as pd
import googleNgram as gn
import wf2020 as wf20 # module for the wright-fisher inspired model
import languageCompute as lc # module for time-series computations such as the cosine-similarity matrices and the BC
seed = 42 # random seeding for reproducibility
from scipy.stats import t

# make directory for precomputed values and simulations
try:
    os.mkdir('save/')
except:
    pass
try:
    os.mkdir('save/google-1gram-computations')
except:
    pass
try:
    os.mkdir('save/other')
except:
    pass
print()

### 1
n = '1'
l_codes = ['eng','eng-us','eng-gb','eng-fiction','chi-sim','fre','ger','ita','heb','rus','spa']
languages = ['English','American English','British English','English Fiction',
             'Simplified Chinese','French','German','Italian','Hebrew','Russian','Spanish']
l_labels = {j:languages[i] for i, j in enumerate(l_codes)}

# data and print data summary table
print('computing data summary table...')
L = {}
L_summary = {'language':[],'c (data)':[],'beta (data)':[],'ln(beta) (data)':[],'c (data) / beta (data)':[],'T (data)':[],'min(r) (data)':[],'max(r) (data)':[]}
for l in l_codes:
    D = gn.read(n,l,ignore_case=True,restriction=True,annotation=False)
    for k in D.keys():
        try:
            D[k] = D[k].T.sort_index().T
        except:
            pass
    L[l] = D
    size = D['rscore'].shape
    L_summary['language'].append(l_labels[l])
    L_summary['c (data)'].append(size[0])
    L_summary['beta (data)'].append(int(np.sum(D['rscore'][1900])))
    L_summary['ln(beta) (data)'].append(round(np.log(np.sum(D['rscore'][1900])),4))
    L_summary['c (data) / beta (data)'].append(size[0]/np.sum(D['rscore'][1900]))
    L_summary['T (data)'].append(size[1])
    L_summary['min(r) (data)'].append(int(np.min(np.min(D['rscore']))))
    L_summary['max(r) (data)'].append(int(np.min(np.max(D['rscore']))))
L_summary = pd.DataFrame(L_summary,index=l_codes)
L_summary.sort_values(by='c (data)')
L_summary.to_csv('save/google-1gram-computations/data-summary.csv')
print()

### 2
print('computing corpus size fits...')
# curve_fit uses a least-squares optimization method (default method='lm' means unconstrained least squares method)
from scipy.optimize import curve_fit 

# corpus size function 
def corpus_size(t,alpha,beta):
    return np.ceil(beta*np.exp(alpha*t))

# log transformed exponential (no ceiling function)
def corpus_size_log(t,alpha,beta):
    return alpha*t + np.log(beta)

# perform the curve fitting, RAM store data and print data summary table
L_corpusSize_fits = {'language':[],'alpha (fit)':[],'alpha (99% CI)':[],'ln(beta) (fit)':[], 'ln(beta) (99% CI)':[],'ln(beta) (data)':[],'c (data)':[],'c (data) / beta (fit)':[],'c (data) / beta (data)':[]}
L_corpusSize_covs = {}

for l in l_codes:
    # time vector as indices
    time_vect = np.array(range(0,L[l]['rscore'].shape[1]))
    # corpus size data
    N = np.log(np.sum(L[l]['rscore']).values)
    # simple curve fitting with initial guesses alpha=0.01 and beta=from_data
    pars, cov = curve_fit(f=corpus_size_log, xdata=time_vect, ydata=N)
    # 99% confidence interval
    alpha = 1.0-0.99
    len_y = len(N) # number of data points
    len_p = len(pars) # number of parameters
    dof = max(0,len_y-len_p) # degrees of freedom
    tval = t.ppf(1.0 - alpha / 2.576, dof) # student-t value for the dof and confidence level
    ci = []
    for i, p,var in zip(range(len_y), pars, np.diag(cov)):
        sigma = var**0.5
        ci.append((p - sigma*tval,p + sigma*tval))
    
    L_corpusSize_fits['language'].append(l_labels[l])
    L_corpusSize_fits['alpha (fit)'].append(round(pars[0],4))
    L_corpusSize_fits['alpha (99% CI)'].append((round(ci[0][0],4),round(ci[0][1],4)))
    L_corpusSize_fits['ln(beta) (fit)'].append(round(np.log(pars[1]),4))
    L_corpusSize_fits['ln(beta) (99% CI)'].append((round(np.log(ci[1][0]),4),round(np.log(ci[1][1]),4)))
    L_corpusSize_fits['ln(beta) (data)'].append(round(np.log(np.sum(L[l]['rscore'][1900])),4))
    L_corpusSize_fits['c (data)'].append(L[l]['rscore'].shape[0])
    L_corpusSize_fits['c (data) / beta (fit)'].append(L[l]['rscore'].shape[0]/int(pars[1]))
    L_corpusSize_fits['c (data) / beta (data)'].append(L[l]['rscore'].shape[0]/int(np.sum(L[l]['rscore'][1900])))
    L_corpusSize_covs[l] = cov
    
L_corpusSize_fits = pd.DataFrame(L_corpusSize_fits,index=l_codes)
L_corpusSize_fits.sort_values(by=['alpha (fit)'])
L_corpusSize_fits.to_csv('save/google-1gram-computations/corpus-size-fits.csv')
np.save('save/google-1gram-computations/corpus-size-covs.npy',L_corpusSize_covs)
print()

### 3
print('computing zipf parameter fits...')

# zipf probabilities (log-log transform) this is the function used to fit the data
def zipf_loglog(x,a,b):
    return (-1*a*x)+b

# perform the curve fitting, RAM store data and print data summary table
L_zipfParameter_fits = {'language':[],'c (data)':[],'a (fit)':[], 'a (99% CI)':[],'b (fit)':[], 'b (99% CI)':[],'E (data)':[],'E (theory)':[]}
L_zipfParameter_covs = {}

for l in l_codes:
    initial_p = L[l]['pscore'][1900]
    empirical_ranks = lc.return_ranks(initial_p)
    er = np.array(list(empirical_ranks.values()),dtype=float)
    er_keys = list(empirical_ranks.keys())
    initial_p_sorted = np.array(list(initial_p.loc[er_keys].values),dtype=float)
    
    # only fit the line using the highest rank up to the empirical expected rank
    E_empirical = lc.zipf_E_data(er,initial_p_sorted)
    E_empirical_round = int(np.floor(E_empirical))
    E_index = np.where(er == E_empirical_round)[0][0]+1
    start_index = 0
    end_index = E_index
    pars, cov = curve_fit(f=zipf_loglog, xdata=np.log(er[start_index:end_index]), ydata=np.log(initial_p_sorted[start_index:end_index]))
    # 99% confidence interval
    alpha = 1-0.99
    len_y = len(np.log(initial_p_sorted[start_index:end_index])) # number of data points
    len_p = len(pars) # number of parameters
    dof = max(0,len_y-len_p) # degrees of freedom
    tval = t.ppf(1.0 - alpha / 2.576, dof) # student-t value for the dof and confidence level
    ci = []
    for i, p,var in zip(range(len_y), pars, np.diag(cov)):
        sigma = var**0.5
        ci.append((p - sigma*tval,p + sigma*tval))

    # theoretical Zipf expected value using the optimal Zipf paramter a and all existing ranks
    ranks = range(1,L[l]['pscore'].shape[0]+1)
    pmf, cdf = wf20.zipf(ranks,pars[0])
    E_theory = wf20.zipf_E(ranks,pars[0]) #np.sum(np.divide(1,np.power(ranks,pars[0]-1)))/np.sum(np.divide(1,np.power(ranks,pars[0])))
    
    L_zipfParameter_fits['language'].append(l_labels[l])
    L_zipfParameter_fits['c (data)'].append(L[l]['pscore'].shape[0])
    L_zipfParameter_fits['a (fit)'].append(round(pars[0],4))
    L_zipfParameter_fits['a (99% CI)'].append((round(ci[0][0],4),round(ci[0][1],4)))
    L_zipfParameter_fits['b (fit)'].append(pars[1])
    L_zipfParameter_fits['b (99% CI)'].append((round(ci[1][0],4),round(ci[1][1],4)))
    L_zipfParameter_fits['E (data)'].append(round(E_empirical,2))
    L_zipfParameter_fits['E (theory)'].append(round(E_theory,2))
    L_zipfParameter_covs[l] = cov
    
L_zipfParameter_fits = pd.DataFrame(L_zipfParameter_fits,index=l_codes)
L_zipfParameter_fits.sort_values(by=['a (fit)'])
L_zipfParameter_fits.to_csv('save/google-1gram-computations/initial-zipf-fits.csv')
np.save('save/google-1gram-computations/initial-zipf-covs.npy',L_zipfParameter_covs)
print()
