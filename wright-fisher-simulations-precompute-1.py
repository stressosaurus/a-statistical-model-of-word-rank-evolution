#!/usr/bin/env python3

print('Initializing...')
import pickle
import os
import numpy as np
import pandas as pd
import wf2020 as wf20 # module for the wright-fisher inspired model
import languageCompute as lc # module for time-series computations
seed = 42 # random seeding for reproducibility

# make directory for precomputed values and simulations
try:
    os.mkdir('save/')
except:
    pass
try:
    os.mkdir('save/wright-fisher-simulations')
except:
    pass
print()
    
### 1.3
# Wright-Fisher model - neutral example
c = 1000 # vocabulary words
a = 1 # Zipf parameter
alpha = 0.01 # corpus size rate of change
beta = 100000 # initial corpus size
T = 109 # total time elapsed (years)

# sampling of the next time step is the word proportions from the current time step
print('computing wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'...')
R, P, Z, S, N = wf20.wright_fisher(c,a,alpha,beta,0,0,T,[],[],0,set_seed=seed,steady=False)
sim_data = {'rscore':R,'pscore':P,'zscore':Z,'sscore':S,'cs':N}
f = open('save/wright-fisher-simulations/wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
pickle.dump(sim_data,f)
f.close()
del R, P, Z, S, N, sim_data
print('saved in save/wright-fisher-simulations as .pkl')
print()

### 1.4
# simulation mean rank and beta varies
t = 0
print('computing zipf mean rank and beta varies at t='+str(t)+'...')
beta_vect = [1000,10000,100000]
for i in beta_vect:
	print('computing wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'...')
	rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(c,a,alpha,i,0,0,T,[],[],0,set_seed=seed)
	sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
	f = open('save/wright-fisher-simulations/wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'.pkl',"wb")
	pickle.dump(sim_data,f)
	f.close()
	del rscore, pscore, zscore, sscore, cs, sim_data
print('saved in save/wright-fisher-simulations as .pkl')
print()

# simulation mean rank and c varies
t = 0
print('computing zipf mean rank and c varies at t='+str(t)+'...')
c_vect = [1000,10000,100000]
for i in c_vect:
	print('computing wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'...')
	rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(i,a,alpha,beta,0,0,T,[],[],0,set_seed=seed)
	sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
	f = open('save/wright-fisher-simulations/wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
	pickle.dump(sim_data,f)
	f.close()
	del rscore, pscore, zscore, sscore, cs sim_data
print('saved in save/wright-fisher-simulations as .pkl')
print()

### 1.5
# binomial simulations with varied beta, c, and time steps
print('computing binomial simulations with varied beta, c, and time steps...')
num_sims = 1000 # number of simulations
beta_vect = [100,300,600] # initial corpus size
c_vect = [3,14,30] # vocabulary sizes
t_vect = [0,25,75] # time snaps
BIN_PROBS = {} # stores the binomial probabilites at the initial time
SIMS_BIN = {} # stores the simulations
print('computing wf simulations with varied c and beta but with different random seeds num_sims='+str(num_sims))
for i in beta_vect:
	BIN_PROBS[i] = {}
	SIMS_BIN[i] = {}
	x_vals = range(0,i+1) # x values for the binomial distribution
	for j in c_vect:
		SIMS_BIN[i][j] = {}
		pmf, cdf = wf20.zipf(range(1,j+1),a) # set Zipf probabilities
		
		# compute binomial probabilities
		P_vects = {}
		for r in range(1,j+1):
			P_vect = []
			index = r-1
			for x in x_vals:
				P_vect.append(wf20.binomial_wf(i,x,pmf[index],j)) # binomial probabilities
			P_vects[r] = np.array(P_vect) # save computations
		BIN_PROBS[i][j] = {'X':x_vals,'P':P_vects}
		
		# simulations - with different random seeds
		SIMS_BIN[i][j] = {t:{} for t in t_vect}
		SIMS = {}
		for k in range(0,num_sims):
			rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(j,a,alpha,i,0,0,T,[],[],0,set_seed=k)
			SIMS[k] = {'rscore':rscore, 'pscore':pscore, 'zscore':zscore, 'sscore':sscore, 'cs':cs}
			for t in t_vect:
				for w in rscore.index:
					try:
						SIMS_BIN[i][j][t][w].append(rscore[t][w])
					except:
						SIMS_BIN[i][j][t][w] = []
			del rscore, pscore, zscore, sscore, cs
		sim_data = SIMS
		f = open('save/wright-fisher-simulations/wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'_sims'+str(num_sims)+'.pkl',"wb")
		pickle.dump(SIMS,f)
		f.close()
		del sim_data
sim_data = {'BIN_PROBS':BIN_PROBS,'SIMS_BIN':SIMS_BIN,'beta_vect':beta_vect,'c_vect':c_vect,'t_vect':t_vect}
f = open('save/wright-fisher-simulations/binomial_c-beta-t-varies.pkl',"wb")
pickle.dump(sim_data,f)
f.close()
del BIN_PROBS, SIMS_BIN, SIMS, sim_data
print('saved in save/wright-fisher-simulations as .pkl')
print()

# fixed and varied parameters
c = 1000 # vocabulary words
a = 1 # Zipf parameter
alpha = 0.01 #0.024 # corpus size rate of change
beta = 100000 # initial corpus size
T = 109 # total time elapsed (years)
bc_all_params = {}
bc_all_simuls = {}
bc_all_labels = {}
bc_all_variab = {}
print('computing wf simulations with varied parameters and computing BC values ...')

# BC - alpha varies
print('BC - alpha varies...')
alpha_vect = [0.01,0.015,0.020,0.025,0.030]
for i in alpha_vect:
    print('computing wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(i)+'_beta'+str(beta)+'_T'+str(T)+'...')
    rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(c,a,i,beta,0,0,T,[],[],0,set_seed=seed,steady=False)
    sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
    f = open('save/wright-fisher-simulations/wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(i)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
    pickle.dump(sim_data,f)
    f.close()
    print('saved in save/wright-fisher-simulations as .pkl')
del rscore, pscore, zscore, sscore, cs, sim_data

# BC - beta varies
print('BC - beta varies...')
beta_vect = [100000,200000,300000,400000,800000]
for i in beta_vect:
    print('computing wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'...')
    rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(c,a,alpha,i,0,0,T,[],[],0,set_seed=seed)
    sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
    f = open('save/wright-fisher-simulations/wf_c'+str(c)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(i)+'_T'+str(T)+'.pkl',"wb")
    pickle.dump(sim_data,f)
    f.close()
    print('saved in save/wright-fisher-simulations as .pkl')
del rscore, pscore, zscore, sscore, cs, sim_data

# BC - c varies
print('BC - c varies...')
c_vect = [1000,2000,3000,4000,8000]
for i in c_vect:
    print('computing wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'...')
    rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(i,a,alpha,beta,0,0,T,[],[],0,set_seed=seed)
    sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
    f = open('save/wright-fisher-simulations/wf_c'+str(i)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
    pickle.dump(sim_data,f)
    f.close()
    print('saved in save/wright-fisher-simulations as .pkl')
del rscore, pscore, zscore, sscore, cs, sim_data

# BC - ratio c/beta = 0.01
print('BC - ratio c/beta = 0.01...')
ratio1_vect = [c_vect[i]/beta_vect[i] for i in range(len(c_vect))]
for i, j in enumerate(c_vect):
    print('computing wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta_vect[i]))+'_T'+str(T)+' with ratio'+str(round(ratio1_vect[i],2))+'...')
    rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(j,a,alpha,beta_vect[i],0,0,T,[],[],0,set_seed=seed)
    sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
    f = open('save/wright-fisher-simulations/wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta_vect[i]))+'_T'+str(T)+'.pkl',"wb")
    pickle.dump(sim_data,f)
    f.close()
    print('saved in save/wright-fisher-simulations as .pkl')
del rscore, pscore, zscore, sscore, cs, sim_data

# BC - ratio c/beta = 0.05
print('BC - ratio c/beta = 0.05...')
beta2_vect = [i/0.05 for i in c_vect]
ratio2_vect = [c_vect[i]/beta2_vect[i] for i in range(len(c_vect))]
for i, j in enumerate(c_vect):
    print('computing wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta2_vect[i]))+'_T'+str(T)+' with ratio'+str(round(ratio2_vect[i],2))+'...')
    rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(j,a,alpha,beta2_vect[i],0,0,T,[],[],0,set_seed=seed)
    sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
    f = open('save/wright-fisher-simulations/wf_c'+str(j)+'_a'+str(a)+'_alpha'+str(alpha)+'_beta'+str(int(beta2_vect[i]))+'_T'+str(T)+'.pkl',"wb")
    pickle.dump(sim_data,f)
    f.close()
    print('saved in save/wright-fisher-simulations as .pkl')
del rscore, pscore, zscore, sscore, cs, sim_data

# BC - Zipf parameter varies
print('BC - Zipf parameter varies...')
a_vect = [0.70,0.80,0.90,1,1.10]
for i in a_vect:
    print('computing wf_c'+str(c)+'_a'+str(i)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'...')
    rscore, pscore, zscore, sscore, cs = wf20.wright_fisher(c,i,alpha,beta,0,0,T,[],[],0,set_seed=seed)
    sim_data = {'rscore':rscore,'pscore':pscore,'zscore':zscore,'sscore':sscore,'cs':cs}
    f = open('save/wright-fisher-simulations/wf_c'+str(c)+'_a'+str(i)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_T'+str(T)+'.pkl',"wb")
    pickle.dump(sim_data,f)
    f.close()
    print('saved in save/wright-fisher-simulations as .pkl')
del rscore, pscore, zscore, sscore, cs, sim_data