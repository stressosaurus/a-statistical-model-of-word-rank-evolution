### Language Plotter
## Alex John Quijano
## Created: 1/26/2018

## Import packages
import os
import re
import sys
import math
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.font_manager as mfm
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
from sklearn.preprocessing import normalize
import languageCompute as lc
from adjustText import adjust_text

#plt.switch_backend('agg')
fontP = mfm.FontProperties(fname='unifont.ttf',size=12)

### Global variables
colors_hex = {  'b':'#0000FF','blue':'#0000FF',
				'g':'#008000','green':'#008000',
				'r':'#FF0000', 'red':'#FF0000',
				'c':'#00FFFF', 'cyan':'#00FFFF',
				'm':'#FF00FF', 'magenta':'#FF00FF',
				'y':'#FFFF00', 'yellow':'#FFFF00',
				'k':'#000000', 'black':'#000000',
				'w':'#FFFFFF', 'white':'#FFFFFF'}

### Plotting utilities

# the function below takes a color as a input (hex-value) and returns its complementary color (hex-value).
def get_complementary(color):
    # strip the # from the beginning
    color = color[1:]
 
    # convert the string into hex
    color = int(color, 16)
 
    # invert the three bytes
    # as good as substracting each of RGB component by 255(FF)
    comp_color = 0xFFFFFF ^ color
 
    # convert the color back to hex by prefixing a #
    comp_color = "#%06X" % comp_color
 
    return comp_color

## To create custom colormaps
def make_colormap(seq):
	
	seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
	cdict = {'red': [], 'green': [], 'blue': []}
	for i, item in enumerate(seq):
		if isinstance(item, float):
			r1, g1, b1 = seq[i - 1]
			r2, g2, b2 = seq[i + 1]
			cdict['red'].append([item, r1, r2])
			cdict['green'].append([item, g1, g2])
			cdict['blue'].append([item, b1, b2])
			
	return cl.LinearSegmentedColormap('CustomMap', cdict)

## To shift a colomap
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
	cdict = {'red': [],'green': [],'blue': [],'alpha': []}
	
	# regular index to compute the colors
	reg_index = np.linspace(start, stop, 257)
	
	# shifted index to match the data
	shift_index = np.hstack([
		np.linspace(0.0, midpoint, 128, endpoint=False), 
		np.linspace(midpoint, 1.0, 129, endpoint=True)
	])
	
	for ri, si in zip(reg_index, shift_index):
		r, g, b, a = cmap(ri)
		cdict['red'].append((si, r, r))
		cdict['green'].append((si, g, g))
		cdict['blue'].append((si, b, b))
		cdict['alpha'].append((si, a, a))
		
	newcmap = cl.LinearSegmentedColormap(name, cdict)
	plt.register_cmap(cmap=newcmap)
	
	return newcmap

### Plotting functions

## time series plot
def time_series_plot(data,xlabel=' ',ylabel=' ',title=' ',color='red',linestyles=['-','--','-.',':','-','--','-','--','-.',':','-','--'],legend=True,annotation=False,annotation_parameters=[],ax=None):
	
	# keys for the ngrams
	ws = data.keys()
	if ws.shape[0] > 12:
		print('Error: Too much word inputs. You can only plot at most 12 time-series on the same figure. \n Automatically plotting only the first 12 words...')
		ws = ws[0:12]
	y_range = data.index
	
	# define colormap given a color
	match_hex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color)
	if not match_hex:
		try:
			color = colors_hex[color]
		except:
			print('Color '+str(color)+' not available. Try converting it to hexadecimal. Using default color...')
			color = colors_hex['red']
	color_complement = get_complementary(color)
	c = cl.ColorConverter().to_rgb
	c_cmap = make_colormap([c(color), c('black'), 0.50, c('black'), c(color_complement)])
	index_color = c_cmap(np.linspace(0,1,len(ws)))
	
	if not ax:
		fig, ax = plt.subplots(figsize=(7.5,3))
	for i, j in enumerate(ws):
		ax.plot(y_range,data[j],color=index_color[i],label=j,linewidth=2,linestyle=linestyles[i])
	if annotation == True:
		params = annotation_parameters
		time_points = params[0]
		time_labels = params[1]
		for tp_index, tp in enumerate(time_points):
			y_min = np.min(np.matrix(data))
			if len(tp) == 1:
				ax.axvline(tp[0],color='gray',alpha=0.50)
				ax.annotate(time_labels[tp_index],xy=(tp[0],y_min),horizontalalignment='center',verticalalignment='center',fontweight='bold',color='black',bbox=dict(boxstyle='square',facecolor='white'),fontsize=12)
			elif len(tp) == 2:
				ax.axvspan(tp[0],tp[1],facecolor='gray',alpha=0.50)
				ax.annotate(time_labels[tp_index],xy=(np.mean(tp),y_min),horizontalalignment='center',verticalalignment='center',fontweight='bold',color='black',bbox=dict(boxstyle='square',facecolor='white'),fontsize=12)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	if legend == True:
		ax.legend(loc=0,prop=fontP)
	else:
		pass
	
	return ax

## plot the pairwise cosine similarity matrix
def pairwise_cosine_similarity_plot(data,data_selected=[],color='red',color_selected='yellow',data_linestyles=['-','--','-.',':'],title=' ',ax=None):
	
	# all data: define colormap given a color
	match_hex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color)
	if not match_hex:
		try:
			color = colors_hex[color]
		except:
			print('Color '+str(color)+' not available. Try converting it to hexadecimal. Using default color...')
			color = colors_hex['red']
	color_complement = get_complementary(color)
	c1 = cl.ColorConverter().to_rgb
	c1_cmap = make_colormap([c1(color), c1('white'), 0.50, c1('white'), c1(color_complement)])
	
	# selected data: define colormap given a color
	match_hex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color_selected)
	if not match_hex:
		try:
			color_selected = colors_hex[color_selected]
		except:
			print('Color '+str(color)+' not available. Try converting it to hexadecimal. Using default color...')
			color_selected = colors_hex['red']
	color_complement_selected = get_complementary(color_selected)
	c2 = cl.ColorConverter().to_rgb
	c2_cmap = make_colormap([c2(color_selected), c2('black'), 0.50, c2('black'), c2(color_complement_selected)])
	index_color = c2_cmap(np.linspace(0,1,len(data_selected)))
	data_index = {j:i for i,j in enumerate(data.keys())}
	
	# draw the matrix as an image
	if not ax:
		fig, ax = plt.subplots(1,1,figsize=(8,6))
	opacity = 1
	if data_selected != []:
		opacity = 0.90
	img = ax.imshow(data,cmap=c1_cmap,vmin=-1,vmax=1,alpha=opacity)
	for i, j in enumerate(data_selected):
		ax.axhline(data_index[j], linewidth=4,color=index_color[i],label=j,linestyle=data_linestyles[i])
	ax.set_xlabel(r'$w_i$',fontsize=14)
	ax.set_ylabel(r'$w_j$',fontsize=14)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_title(title)
	if data_selected != []:
		ax.legend(loc=0,prop=fontP)
	cbar = plt.colorbar(img,orientation='vertical',ax=ax)
	cbar.set_label('cosine similarity')
	
	return ax

## pairwise cosine distribution plot
def cosine_distribution_plot(data,linestyles=['-','--','-.',':'],color='red',title=' ',bins=50,ax=None):
	
	# keys for the ngrams
	ws = data.keys()
	if ws.shape[0] > 4:
		print('Error: Too much word inputs. You can only plot at most three cosine distributions on the same figure. \n Automatically plotting only the first four words...')
		ws = ws[0:3]
	
	# define colormap given a color
	match_hex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color)
	if not match_hex:
		try:
			color = colors_hex[color]
		except:
			print('Color '+str(color)+' not available. Try converting it to hexadecimal. Using default color...')
			color = colors_hex['red']
	color_complement = get_complementary(color)
	c = cl.ColorConverter().to_rgb
	c_cmap = make_colormap([c(color), c('black'), 0.50, c('black'), c(color_complement)])
	index_color = c_cmap(np.linspace(0,1,len(ws)))
	
	if not ax:
		fig, ax = plt.subplots(figsize=(7.5,3))
	for i, j in enumerate(ws):
		ax.hist(data[j],bins=bins,color=index_color[i],label=j,histtype='step',linewidth=2,linestyle=linestyles[i])
		ax.hist(data[j],bins=bins,color=index_color[i],alpha=0.10)
	ax.set_xlabel('cosine similarity')
	ax.set_ylabel('density')
	ax.set_title(title)
	ax.legend(loc=0,prop=fontP)
	
	return ax

## plot bimodality coefficient and data
def bimodality_coefficient_plot(data={},data_selected=[],domain=[-2,-2,2,2],grid_size=5000,title=' ',color='red',color_selected='yellow',markerstyle='.',markerstyle_selected=['o','s','^','x'],markersize=5,markersize_selected=6,cmap_selected=True,annotation_selected=False,linewidth=2,legend=True,colorbar=True,ax=None):
	
	# check if data inputs are valid
	if data_selected != [] and data == {}:
		print('Error: Input data is not valid! the \'data\' parameter needs to be defined if \'data_selected\' is nonzero.')
		data_selected = []
	if len(data_selected) > 4:
		print('Error: Too much word inputs. You can only plot at most three words on the same figure. \n Automatically plotting only the first four words...')
		data_selected = data_selected[0:4]
	
	# define default domains
	m4_b = [domain[0],domain[2]]
	m3_b = [domain[1],domain[3]]
	
	# define critical line
	crit = 5.0/9.0 # critical line 5/9
	m3_crit = np.linspace(m3_b[0],m3_b[1],grid_size)
	m4_crit = lc.bc_curve_m3(m3_crit,crit,n=grid_size)
	crit_label = r'$BC_{critical} = \frac{5}{9}$'
	
	# define normal line
	norm = 1.0/3.0 # normal line 1/3
	m3_norm = np.linspace(m3_b[0],m3_b[1],grid_size)
	m4_norm = lc.bc_curve_m3(m3_norm,norm,n=grid_size)
	norm_label = r'$BC_{normal} = \frac{1}{3}$'
	
	# define bc space gradient
	m4 = np.linspace(m4_b[0],m4_b[1],grid_size) # kurtosi0
	m3 = np.linspace(m3_b[0],m3_b[1],grid_size) # skewness
	m4_X, m3_Y = np.meshgrid(m4,m3)
	BC_vals = lc.sample_bimodality_coefficient(m4_X,m3_Y)
	dx = (m4[1]-m4[0])/2 # x-axis is kurtosis
	dy = (m3[1]-m3[0])/2 # y-axis is skewness
	extent = [m4[0]-dx, m4[-1]+dx, m3[0]-dy, m3[-1]+dy]
	
	# define colormaps and lines
	midpoint = crit/np.max(BC_vals) - 0.03 # colormap midpoint
	m4_lines = [m4_crit,m4_norm]
	m3_lines = [m3_crit,m3_norm]
	lines_label = [crit_label,norm_label]
	lines_color = ['blue','orange']
	orig_cmap = cm.Greys
	shifted_cmap = shiftedColorMap(orig_cmap, midpoint=midpoint, name='shifted')
	
	# selected data: define colormap given a color
	if cmap_selected == True:
		match_hex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color_selected)
		if not match_hex:
			try:
				color_selected = colors_hex[color_selected]
			except:
				print('Color '+str(color)+' not available. Try converting it to hexadecimal. Using default color...')
				color_selected = colors_hex['red']
		color_complement_selected = get_complementary(color_selected)
		c2 = cl.ColorConverter().to_rgb
		c2_cmap = make_colormap([c2(color_selected), c2('black'), 0.50, c2('black'), c2(color_complement_selected)])
		index_color = c2_cmap(np.linspace(0,1,len(data_selected)))
	elif cmap_selected == False:
		index_color = color_selected
	
	# set-up dataset to plot
	if data != {}:
		m3_data = data['m3']
		m4_data = data['m4']
	
	# draw gradients, lines, and datapoints
	if not ax:
		fig, ax = plt.subplots(1,1,figsize=(7,4.666))
	img = ax.imshow(BC_vals, extent=extent, origin='lower', cmap=shifted_cmap)
	for k in range(len(m4_lines)):
		ax.plot(m4_lines[k],m3_lines[k],'--',label=lines_label[k],color=lines_color[k],linewidth=linewidth)
	if data != {}:
		for i, j in enumerate(m3_data):
			ax.plot(m4_data[i],j,color=color,marker=markerstyle,markersize=markersize,linestyle='None')
	if data_selected != []:
		for i, j in enumerate(data_selected):
			if annotation_selected == False:
				ax.plot(m4_data[j],m3_data[j],marker=markerstyle_selected[i],markersize=markersize_selected,color=index_color[i],label=j,linestyle='None')
			elif annotation_selected == True:
				ax.plot(m4_data[j],m3_data[j],markersize=markersize_selected)
				ax.annotate(j,xy=(m4_data[j],m3_data[j]),xytext=(m4_data[j]+0.05,m3_data[j]+0.05))
	ax.set_xlim(m4_b)
	ax.set_ylim(m3_b)
	ax.set_xlabel('kurtosis')
	ax.set_ylabel('skewness')
	ax.set_title(title)
	if legend == True:
		ax.legend(loc='upper center',prop=fontP,ncol=3)
	if colorbar == True:
		cbar = plt.colorbar(img,orientation='vertical',ax=ax)
		cbar.set_label('bimodality coefficient (BC)')
		
	return ax

## plot the word relations based on a relation matrix and plot the time-series
def word_relation_time_series_plot(w,df,ts,top=3,bottom=3,ax=None,title=''):
    # get related words
    t_vect = list(reversed(ts.columns))
    related_w_top = pd.DataFrame(df[w]).sort_values(by=[w],ascending=False)[0:top+1]
    related_w_bottom = pd.DataFrame(df[w]).sort_values(by=[w],ascending=True)[0:bottom].sort_values(by=[w],ascending=False)
    related_w = pd.concat([related_w_top,related_w_bottom]).T
    c = []
    h = []
    col = []
    for i in list(related_w.keys()):
        weight_val = round(related_w[i],4).values[0]
        h.append(weight_val)
        if weight_val == 1:
            col.append('black')
        elif weight_val >= 0:
            col.append('blue')
        else:
            col.append('red')
        c.append(i)
    
    if not ax:
        fig, ax = plt.subplots(figsize=(5,5))
        
    axs = ts.loc[c].T.plot(subplots=True,legend=False,yticks=[],xticks=[],color=col,grid=True,sharex=True,ax=ax)
    
    for j, i in enumerate(axs):
        if j == 0:
            i.set_title(title)
            i.annotate(c[j],xy=(t_vect[0],0),xytext=(t_vect[0]+2,0),color=col[j],verticalalignment='center',fontweight='bold')
        else:
            i.annotate(c[j],xy=(t_vect[0],0),xytext=(t_vect[0]+2,0),color=col[j],verticalalignment='center')
        i.plot(t_vect,[0]*len(t_vect),'--',color='gray',alpha=0.5)
        i.axis('off')
        
    return axs

## plot the word relations based on a relation matrix
def word_relation_network_plot(w,df,top=3,bottom=3,ax=None,title=''):
    # get related words
    related_w_top = pd.DataFrame(df[w]).sort_values(by=[w],ascending=False)[0:top+1]
    related_w_bottom = pd.DataFrame(df[w]).sort_values(by=[w],ascending=True)[0:bottom]
    related_w = pd.concat([related_w_top,related_w_bottom]).T
    edges = {}
    edges_col = []
    for i in related_w.keys():
        weight_val = round(related_w[i],4).values[0]
        edges[(w,i)] = weight_val
        if weight_val >= 0:
            edges_col.append('black')
        else:
            edges_col.append('red')
            
    # create circle for drawing
    r = 0.5
    xy = []
    for i in edges.keys():
        theta = np.arccos(edges[i])
        r = 1
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        xy.append((x,y,i[1]))

    if not ax:
        fig, ax = plt.subplots(figsize=(6.5,5))
        
    text = [w for (x,y,w) in xy]
    eucs = [x for (x,y,w) in xy]
    covers = [y for (x,y,w) in xy]

    # plotting the lines
    ax.plot(eucs,covers,alpha=0)
    texts = []
    for x, y, s in zip(eucs, covers, text):
        if x == 1.0 and y == 0.0:
            ax.plot((0,x),(0,y),'-',color='black',alpha=1,linewidth=3)
            texts.append(ax.annotate(s,(x,y),color='black'))
        elif x >= 0.0 and x < 1.0:
            ax.plot((0,x),(0,y),'-',color='blue',alpha=0.25)
            texts.append(ax.annotate(s,(x,y),color='blue'))
        elif x >= -1 and x < 0.0 :
            ax.plot((0,x),(0,y),'-',color='red',alpha=0.25)
            texts.append(ax.annotate(s,(x,y),color='red'))
            
        #texts.append(ax.annotate(s,(x,y)))
    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='k', lw=1), ax=ax)    
    ax.set_title(title)
    ax.axis('off')
    
    return ax