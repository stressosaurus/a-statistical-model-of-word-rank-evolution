### Language models and functions
## Alex John Quijano
## Created: March 2020

import pandas as pd
import numpy as np
import math
from scipy.special import comb

## Zipf pmf and cdf
def zipf(k,a):
	power_law = np.divide(1,np.power(k,a))
	H = np.sum(np.divide(1,np.power(k,a)))
	pmf = np.divide(power_law,H)
	cdf = [np.sum(pmf[0:i]) for i in range(1,pmf.shape[0]+1)]
	return pmf, cdf

## Zipf expected rank/value - theory
def zipf_E(k,a):
	return np.sum(np.divide(1,np.power(k,a-1)))/np.sum(np.divide(1,np.power(k,a)))

## Zipf log-log transform
def zipf_loglog(x,a,b):
    return (-1*a*x)+b

## Vocabulary construction where c is the number of unigrams
def construct_vocabulary(c):
	return [int(i) for i in range(1,c+1,1)]

## Wright-Fisher model - Corpus size function
def corpus_size(t,alpha,beta):
	return int(np.ceil(beta*np.exp(alpha*t)))

## Wright-Fisher model - Selection value function
def selection_function(t,A,B):
	return A*(np.sin(B*t) + np.cos(B*t))

# Wright-Fisher model (with selection option)
def wright_fisher(c,a,alpha,beta,A,B,T,PV,NV,tau,set_seed=None,steady=False):
	
	# parameters - descriptions
	# 1.  c     - is the number of vocabulary words
	# 2.  a     - is the Zipf parameter
	# 3.  alpha - is the rate of change for the corpus size
	# 4.  beta  - is the initial corpus size
	# 5.  A     - is the selection value
	# 6.  B     - is the variable selection value
	# 7.  T     - is the total time elapsed (number of years)
	# 8.  PV    - word indices under positive selection
	# 9.  NV    - word indices under negative selection
	# 10. tau  - the time where selection is induced
	
	# seeding
	np.random.seed(set_seed)
	
	# vocabulary and ranks ranks
	V = construct_vocabulary(c=c) # ngrams labels
	ranks = V # assign ranks to the words
	
	# initial conditions
	t = 0
	pmf, cdf = zipf(k=ranks,a=a) # initial probability distibution of words
	initial_cs = corpus_size(t,alpha,beta) # initial corpus size (exponential function with parameters beta and alpha)
	
	# initial fitness values
	initial_fv = np.zeros(len(V)) + 1
	if tau == 0:
		initial_fv[[i-1 for i in PV]] = 1 + selection_function(tau,A,B)
		initial_fv[[i-1 for i in NV]] = 1 - selection_function(tau,A,B)
	fv_probs = np.multiply(pmf,initial_fv)
	fv_probs_normal = np.divide(fv_probs,np.sum(fv_probs)) # update probabilities
	
	if steady == True:
		fv_probs_normal = pmf
		
	# initial word counts
	word_samples = V # sample words at least once
	word_samples = np.append(word_samples,np.random.choice(V,initial_cs-c,replace=True,p=fv_probs_normal))
	wp_u, wp_c = np.unique(word_samples,return_counts=True)
	del word_samples
	initial_count = np.zeros(len(V))
	initial_count[[i-1 for i in wp_u]] = wp_c
	initial_probs = np.zeros(len(V))
	initial_probs[[i-1 for i in wp_u]] = np.divide(wp_c,np.sum(wp_c))
	
	# time loop
	fv_track = [initial_fv]
	cs_track = [initial_cs]
	count_track = [initial_count]
	probs_track = [initial_probs]
	for i in range(1,T):
		
		# selection at t >= tau
		fv = np.zeros(len(V)) + 1
		if i >= tau:
			fv[[i-1 for i in PV]] = 1 + selection_function(i,A,B)
			fv[[i-1 for i in NV]] = 1 - selection_function(i,A,B)
			fv_track.append(fv)
		else:
			fv_track.append(fv)
		fv_probs = np.multiply(probs_track[i-1],fv_track[i-1])
		fv_probs_normal = np.divide(fv_probs,np.sum(fv_probs)) # update probabilities
		
		if steady == True:
			fv_probs_normal = pmf
			
		# Wright-Fisher
		cs_track.append(corpus_size(i,alpha,beta)) # update corpus size
		word_samples = V # sample words at least once
		word_samples = np.append(word_samples,np.random.choice(V,cs_track[i]-c,replace=True,p=fv_probs_normal))
		wp_u, wp_c = np.unique(word_samples,return_counts=True)
		del word_samples
		next_count = np.zeros(len(V))
		next_count[[i-1 for i in wp_u]] = wp_c # update counts
		count_track.append(next_count)
		next_probs = np.zeros(len(V))
		next_probs[[i-1 for i in wp_u]] = wp_c/np.sum(wp_c) # update probabilities
		probs_track.append(next_probs)
		
	# compute pscores, zscores, and convert outputs to dataframes
	R = pd.DataFrame(np.matrix(count_track).T,index=V,columns=range(0,T))
	P = pd.DataFrame(np.divide(R,np.sum(R,axis=0)),index=V,columns=range(0,T))
	a = P.T - np.mean(P.T,axis=0)
	b = np.std(P.T,axis=0)
	Z = np.divide(a,b).T
	S = pd.DataFrame(np.matrix(fv_track).T,index=V,columns=range(0,T))
	N = pd.DataFrame({'N(t)':cs_track},index=R.columns)
	del fv_track, cs_track, count_track, probs_track
	
	return R, P, Z, S, N

# Wright-Fisher model (with selection option and zero words)
def wright_fisher_0(c,a,alpha,beta,A,B,T,PV,NV,tau,set_seed=None,steady=False):
	
	# parameters - descriptions
	# 1.  c     - is the number of vocabulary words
	# 2.  a     - is the Zipf parameter
	# 3.  alpha - is the rate of change for the corpus size
	# 4.  beta  - is the initial corpus size
	# 5.  A     - is the selection value
	# 6.  B     - is the variable selection value
	# 7.  T     - is the total time elapsed (number of years)
	# 8.  PV    - word indices under positive selection
	# 9.  NV    - word indices under negative selection
	# 10. tau  - the time where selection is induced
	
	# seeding
	np.random.seed(set_seed)
	
	# vocabulary and ranks ranks
	V = construct_vocabulary(c=c) # ngrams labels
	ranks = V # assign ranks to the words
	
	# initial conditions
	t = 0
	pmf, cdf = zipf(k=ranks,a=a) # initial probability distibution of words
	initial_cs = corpus_size(t,alpha,beta) # initial corpus size (exponential function with parameters beta and alpha)
	
	# initial fitness values
	initial_fv = np.zeros(len(V)) + 1
	if tau == 0:
		initial_fv[[i-1 for i in PV]] = 1 + selection_function(tau,A,B)
		initial_fv[[i-1 for i in NV]] = 1 - selection_function(tau,A,B)
	fv_probs = np.multiply(pmf,initial_fv)
	fv_probs_normal = np.divide(fv_probs,np.sum(fv_probs)) # update probabilities
	
	if steady == True:
		fv_probs_normal = pmf
		
	# initial word counts
	word_samples = np.random.choice(V,initial_cs,replace=True,p=fv_probs_normal)
	wp_u, wp_c = np.unique(word_samples,return_counts=True)
	del word_samples
	initial_count = np.zeros(len(V))
	initial_count[[i-1 for i in wp_u]] = wp_c
	initial_probs = np.zeros(len(V))
	initial_probs[[i-1 for i in wp_u]] = np.divide(wp_c,np.sum(wp_c))
	
	# time loop
	fv_track = [initial_fv]
	cs_track = [initial_cs]
	count_track = [initial_count]
	probs_track = [initial_probs]
	for i in range(1,T):
		
		# selection at t >= tau
		fv = np.zeros(len(V)) + 1
		if i >= tau:
			fv[[i-1 for i in PV]] = 1 + selection_function(i,A,B)
			fv[[i-1 for i in NV]] = 1 - selection_function(i,A,B)
			fv_track.append(fv)
		else:
			fv_track.append(fv)
		fv_probs = np.multiply(probs_track[i-1],fv_track[i-1])
		fv_probs_normal = np.divide(fv_probs,np.sum(fv_probs)) # update probabilities
		
		if steady == True:
			fv_probs_normal = pmf
			
		# Wright-Fisher
		cs_track.append(corpus_size(i,alpha,beta)) # update corpus size
		word_samples = np.random.choice(V,cs_track[i],replace=True,p=fv_probs_normal)
		wp_u, wp_c = np.unique(word_samples,return_counts=True)
		del word_samples
		next_count = np.zeros(len(V))
		next_count[[i-1 for i in wp_u]] = wp_c # update counts
		count_track.append(next_count)
		next_probs = np.zeros(len(V))
		next_probs[[i-1 for i in wp_u]] = wp_c/np.sum(wp_c) # update probabilities
		probs_track.append(next_probs)
		
	# compute pscores, zscores, and convert outputs to dataframes
	R = pd.DataFrame(np.matrix(count_track).T,index=V,columns=range(0,T))
	P = pd.DataFrame(np.divide(R,np.sum(R,axis=0)),index=V,columns=range(0,T))
	a = P.T - np.mean(P.T,axis=0)
	b = np.std(P.T,axis=0)
	Z = np.divide(a,b).T
	S = pd.DataFrame(np.matrix(fv_track).T,index=V,columns=range(0,T))
	N = pd.DataFrame({'N(t)':cs_track},index=R.columns)
	del fv_track, cs_track, count_track, probs_track
	
	return R, P, Z, S, N

# binomial (pmf)
def binomial(n,x,p,type='pmf'):
	if x >= 0 and x <= n:
		if type == 'pmf':
			f = comb(n,x)*np.power(p,x)*np.power(1-p,n-x)
		elif type == 'cdf':
			f = 0
			for i in range(0,x+1):
				f += comb(n,i)*np.power(p,i)*np.power(1-p,n-i)
	else:
		f = 0
	return f

# binomal (pmf)
def binomial_wf(n,x,p,c):
	
	if x <= n-c:
		p = binomial(n-c,x,p)
	elif x > n-c:
		p = 0
		
	return p

# binomial expected value
def E(n,p):
	return n*p

# binomial variance
def Var(n,p):
	return n*p*(1-p)

# binomial skewness
def Skew(n,p):
	return np.divide((1-p)-p,np.sqrt(n*p*(1-p)))

# binomial covariance
def Cov(n,p_i,p_j):
	return -1*n*p_i*p_j