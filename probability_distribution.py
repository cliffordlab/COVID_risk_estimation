# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 01:12:20 2021

@author: chait
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def path_length(weights, mci_mu, mci_sig):
    w_mci = weights[0]
    w_healthy = weights[1]
    locg = mci_mu[0]
    scaleg = mci_sig[0]
    locl = mci_mu[1]
    scalel = mci_sig[1]
    # Get gaussian RV, i.e. sample from Gaussian dist
    gaussian = stats.norm(loc=locg,scale=scaleg)
    gaussian_rv = gaussian.rvs(size=1)[0]
    while gaussian_rv <= 0:
        gaussian_rv = gaussian.rvs(size=1)[0]
    
    # Get levy RV, i.e. sample from Levy dist
    levy = stats.levy(loc=locl,scale=scalel)
    levy_rv = levy.rvs(size=1)[0]
    
    # Pick gaussian RV with prob w_mci or levy RV with prob w_healthy
    dist_choice = np.random.choice(['g', 'l'], p=[w_mci, w_healthy])[0]
    if dist_choice == 'g':
        #print(dist_choice)
        return int(gaussian_rv)
    else:
        #print(dist_choice)
        return int(levy_rv)
    
    
#################################################### Displaying PDFs #######################################################
# NOTE: Uncomment to see graphs of the PDFs
        
## Levy dist. Ref: https://www.geeksforgeeks.org/python-levy-distribution-in-statistics/
#xl = np.linspace(0,50,200)
#yl = stats.levy.pdf(xl,20,5)
#plt.plot(xl,yl)
#plt.title('Levy Distribution PDF')
#plt.xlabel('Path length')
#plt.ylabel('Probability of path length to be picked')
#
## Gaussian distribution. Ref: https://stackoverflow.com/questions/10138085/python-plot-normal-distribution
#xg = np.linspace(0,50,200)
#yg = stats.norm.pdf(xg,10,5)
#plt.plot(xg,yg)
#plt.title('Gaussian distribution pdf')
#
## Histogram for combined distribution
#step_lengths = []
#for i in range(100000):
#    step_lengths.append(path_length((0.5,0.5), (14,20), (5,5)))
#    
#count = np.zeros(len(np.unique(step_lengths)))
#num_bins = len(np.unique(step_lengths))
#for i in range(len(step_lengths)):
#    for j in range(len(count)):
#        if step_lengths[i] == j:
#            count[j] += 1
#count = count/len(step_lengths)
#
#plt.plot(count, label='Mixed dist')
#plt.plot(xl,yl, label='Levy')
#plt.plot(xg,yg, label='Gaussian')
#plt.legend()
#plt.xlabel('Path length (in # of pixels)')
#plt.ylabel('Probability of path length to be picked')