### Language Computations
## Alex John Quijano
## Created: 2/18/2018

import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import norm

#### General Utilities ####

## zipf expected rank/value - empirical
def zipf_E_data(er,df):
	return np.sum(np.multiply(er,df))

## returns (output is word-pairs) the top n cosine similar ngrams for a given list of ngrams
def cc_top(grams,grams_cosines,vocabulary,n=5,k=0):
	
	pairs = []
	for j, i in enumerate(grams_cosines):
		i_sorted_index = np.squeeze(np.array(np.argsort(-1*i)))[0:n-1]
		for c in i_sorted_index:
			pairs.append((grams[j],vocabulary['reverse'][c]))
			
	return np.array(pairs)

## construct cscore (word classes - needs existing text files)
def readClasses(n,g,l,vocabulary):
	
	list_class = []
	file = open(n+'gram-list/'+g+'/'+n+'gram-class-'+l)
	for f in file:
		list_class.append(f.replace('\n',''))
	list_gram_all = {}
	list_gram_exist = {}
	cscore = np.zeros((len(vocabulary['forward']),len(list_class)),dtype=int)
	for lc_i, lc in enumerate(list_class):
		list_gram_all[lc] = []
		list_gram_exist[lc] = []
		file_1 = open(n+'gram-list/'+g+'/'+l+'/'+n+'gram-list-'+lc+'-'+l,'r')
		for f in file_1:
			gram = f.replace('\n','')
			list_gram_all[lc].append(gram)
			try:
				gram_indexInVocabulary = vocabulary['forward'][gram]
				list_gram_exist[lc].append(gram)
				cscore[gram_indexInVocabulary,lc_i] = 1
			except KeyError:
				pass
		file_1.close()
		
	return cscore, np.array(list_class), list_gram_exist, list_gram_all

# generate pscore matrix
def generate_pscore(rscore_matrix):
	
	return np.divide(rscore_matrix,np.sum(rscore_matrix,axis=0))

# generate zscore matrix
def generate_zscore(pscore_matrix):
	
	means = np.mean(pscore_matrix,axis=1)
	stds = np.std(pscore_matrix,axis=1)
	out = pd.DataFrame(np.zeros(pscore_matrix.shape),index=pscore_matrix.index,columns=pscore_matrix.columns)
	for i in out.columns:
		out[i] = pscore_matrix[i] - means
	for j in out.columns:
		out[j] = np.divide(out[j],stds)
	
	return out

# generate delta matrix
def generate_delta(data_matrix):
	
	t_vect = data_matrix.columns
	delta = pd.DataFrame(np.zeros(data_matrix.shape,dtype=int),index=data_matrix.index,columns=data_matrix.columns)
	for t in t_vect[:-1]:
		diff = data_matrix[t+1]-data_matrix[t]
		delta[t+1] = diff
	delta = delta.T.drop([t_vect[0]]).T
	
	return delta

#### Ranking computations ####

## assign word rank using data (probabilty scores or raw scores as inputs)
def return_ranks(df):
	df_sort = df.sort_index().sort_values(ascending=False) #np.argsort(-1*df.values)
	ranks = {}
	for i, j in enumerate(df_sort.keys()):
		ranks[j] = i+1
	
	return ranks

#### Similarity Computations ####

## compute pairwise cosine similarity matrix
def pairwise_cosine_similarity(df):
	return pd.DataFrame(cosine_similarity(df),index=df.index,columns=df.index)

## rth sample moment function
def sample_moment(df,r):
	return (1/df.shape[0])*np.sum(np.power(df-np.mean(df),r))

## sample skewness function
def sample_skewness(df):
	return np.divide(sample_moment(df,3),np.power(np.var(df,ddof=1),3/2))

## sample kurtosis function
def sample_kurtosis(df):
	return np.divide(sample_moment(df,4),np.power(np.var(df),2))-3

## sample bc function
def sample_bimodality_coefficient(m4,m3):
	
	n = m3.shape[0]
	b = float(np.power(n-1,2))/float((n-2)*(n-3))
	bc = (np.power(m3,2) + 1)/(m4+3*b)
	
	return bc

## Compute CC matrix and BC values of a given data matrix
def bimodality_coefficient(df,pcs=True):
	
	M = df
	if pcs == True:
		M = pd.DataFrame(np.ma.masked_values(np.matrix(pairwise_cosine_similarity(df)),1),index=df.index,columns=df.index)
	elif pcs == False:
		M = df
	m3 = sample_skewness(M)
	m4 = sample_kurtosis(M)
	bc = sample_bimodality_coefficient(m4,m3)
	out = {'m3':m3,'m4':m4,'bc':bc}
	del M, m3, m4, bc
	
	return out

## BC - Critical line m4 <- f(m3)
def bc_curve_m3(m3,c,n=1000):
	
	if isinstance(m3,int):
		N = n
	else:
		N = m3.shape[0]
	b = float(np.power(N-1,2))/float((N-2)*(N-3))
	a = ((1/c)*(np.power(m3,2)+1))-3*b
	
	return a

## BC - Critical line m3 <- f(m4)
def bc_curve_m4(m4,c,n=1000):
	
	if isinstance(m4,int):
		N = n
	else:
		N = m4.shape[0]
	b = float(np.power(N-1,2))/float((N-2)*(N-3))
	m3 = np.sqrt(c*(m4+3*b)-1)
	
	return m3

#### Statistical Tests ####

## Mann-Kendall test
def mk_test(x, alpha=0.05):
	"""
	Credit: (see LICENCE)
	Created on Wed Jul 29 09:16:06 2015
	@author: Michael Schramm (Github)
	"""

	n = len(x)

	# calculate S
	s = 0
	for k in range(n-1):
			for j in range(k+1, n):
					s += np.sign(x[j] - x[k])

	# calculate the unique data
	unique_x = np.unique(x)
	g = len(unique_x)

	# calculate the var(s)
	if n == g:  # there is no tie
			var_s = (n*(n-1)*(2*n+5))/18
	else:  # there are some ties in data
			tp = np.zeros(unique_x.shape)
			for i in range(len(unique_x)):
					tp[i] = sum(x == unique_x[i])
			var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

	if s > 0:
			z = (s - 1)/np.sqrt(var_s)
	elif s == 0:
			z = 0
	elif s < 0:
			z = (s + 1)/np.sqrt(var_s)

	# calculate the p_value
	p = 2*(1-norm.cdf(abs(z)))  # two tail test
	h = abs(z) > norm.ppf(1-alpha/2)

	if (z < 0) and h:
			trend = 'decreasing'
	elif (z > 0) and h:
			trend = 'increasing'
	else:
			trend = 'neither'

	return trend, h, p, z