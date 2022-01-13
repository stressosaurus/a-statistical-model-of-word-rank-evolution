#!/usr/bin/env python3

print('Initializing...')
import pickle as pkl
import os
import numpy as np
import pandas as pd
import googleNgram as gn
import wf2020 as wf20 # module for the wright-fisher inspired model
import languageCompute as lc # module for time-series computations
from scipy.stats import variation

# make directory for precomputed values and simulations
try:
    os.mkdir('save/')
except:
    pass
try:
    os.mkdir('save/rank-change-computations')
except:
    pass

# directory for the precomputed values and simulations
precomp_gn_dir = 'save/google-1gram-computations'
precomp_wf_dir = 'save/wright-fisher-simulations'

# Languages
print('ranking languages...')
n = '1'
l_codes = ['eng','eng-us','eng-gb','eng-fiction','chi-sim','fre','ger','ita','heb','rus','spa']
for l in l_codes:
    print(l+'...')
    D = gn.read(n,l,ignore_case=True,restriction=True,annotation=False)
    for k in D.keys():
        try:
             D[k] = D[k].T.sort_index().T
        except:
             pass
    P = D['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_'+l+'.pkl...')
    RANKS_LA = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_'+l+'.pkl',"wb")
    pkl.dump(RANKS_LA,f)
    f.close()

# Wright-Fisher Simulations
print('ranking Wright-Fisher simulations...')
# fixed and varied parameters
c = 1000 # vocabulary words
a = 1 # Zipf parameter
alpha = 0.01 #0.024 # corpus size rate of change
beta = 100000 # initial corpus size
T = 109 # total time elapsed (years)

# alpha varies
print('alpha varies...')
alpha_vect = [0.01,0.015,0.020,0.025,0.030]
for i in alpha_vect:
    pca_pre_sim = pkl.load(open(precomp_wf_dir+'/wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(i)+'_beta'+str(beta)+'_T'+str(T)+'.pkl','rb'))
    P = pca_pre_sim['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(i)+'_beta'+str(beta)+'_T'+str(T)+'.pkl...')
    RANKS_WF = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(i)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
    pkl.dump(RANKS_WF,f)
    f.close()

# beta varies
print('beta varies...')
beta_vect = [100000,200000,300000,400000,800000]
for i in beta_vect:
    pca_pre_sim = pkl.load(open(precomp_wf_dir+'/wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'.pkl','rb'))
    P = pca_pre_sim['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'.pkl...')
    RANKS_WF = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'.pkl',"wb")
    pkl.dump(RANKS_WF,f)
    f.close()
    
# c varies
print('c varies...')
c_vect = [1000,2000,3000,4000,8000]
for i in c_vect:
    pca_pre_sim = pkl.load(open(precomp_wf_dir+'/wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl','rb'))
    P = pca_pre_sim['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl...')
    RANKS_WF = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
    pkl.dump(RANKS_WF,f)
    f.close()

# ratio c/beta = 0.01
print('ratio c/beta = 0.01...')
ratio1_vect = [c_vect[i]/beta_vect[i] for i in range(len(c_vect))]
for i, j in enumerate(c_vect):
    pca_pre_sim = pkl.load(open(precomp_wf_dir+'/wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta_vect[i]))+'_T'+str(T)+'.pkl','rb'))
    P = pca_pre_sim['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta_vect[i]))+'_T'+str(T)+'.pkl...')
    RANKS_WF = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta_vect[i]))+'_T'+str(T)+'.pkl',"wb")
    pkl.dump(RANKS_WF,f)
    f.close()

# ratio c/beta = 0.05
print('ratio c/beta = 0.05...')
beta2_vect = [i/0.05 for i in c_vect]
ratio2_vect = [c_vect[i]/beta2_vect[i] for i in range(len(c_vect))]
for i, j in enumerate(c_vect):
    pca_pre_sim = pkl.load(open(precomp_wf_dir+'/wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta2_vect[i]))+'_T'+str(T)+'.pkl','rb'))
    P = pca_pre_sim['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta2_vect[i]))+'_T'+str(T)+'.pkl...')
    RANKS_WF = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta2_vect[i]))+'_T'+str(T)+'.pkl',"wb")
    pkl.dump(RANKS_WF,f)
    f.close()

# Zipf parameter varies
print('Zipf parameter varies...')
a_vect = [0.70,0.80,0.90,1,1.10]
for i in a_vect:
    pca_pre_sim = pkl.load(open(precomp_wf_dir+'/wf_c'+str(c)+'_a'+str(i)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl','rb'))
    P = pca_pre_sim['pscore']
    T = P.shape[1]
    t_vect = P.columns
    ranks = pd.DataFrame(np.zeros(P.shape,dtype=int),index=P.index,columns=P.columns)
    olist = pd.DataFrame(np.zeros(P.shape,dtype=str),index=range(1,P.shape[0]+1),columns=P.columns)
    for t in t_vect:
        val = lc.return_ranks(P[t])
        for ii, jj in val.items():
            ranks[t][ii] = jj
            olist[t][jj] = ii
    dranks = lc.generate_delta(ranks)
    rc_la = {'drank-variance':{},'drank-sum':{}}
    drank_sum = np.sum(dranks,axis=1)
    drank_var = np.var(dranks,axis=1)
    # save computations
    print('saving rank-change-computations/ranks_wf_c'+str(c)+'_a'+str(i)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl...')
    RANKS_WF = {'ranks':ranks,'dranks':dranks,'olist':olist,'drank-variance':drank_var,'drank-sum':drank_sum}
    f = open('save/rank-change-computations/ranks_wf_c'+str(c)+'_a'+str(i)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
    pkl.dump(RANKS_WF,f)
    f.close()