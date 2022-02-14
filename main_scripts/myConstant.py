# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:35:02 2019

@author: Huanyu Kuo
MEE_Constant.py
"""
#
# FILE READING AND SETTINGS
#
InputFileDir = '../input/'
OutputFileDir = '../output/'

NUMBER_OF_PROCESSES = 16 # Multi-processing
#
# To know the number of CPU core, execute:
# import psutil
# num_cpus = psutil.cpu_count(logical=False)
# print('CPU count:', num_cpus)
# OR
# import multiprocessing
# num_process = multiprocessing.cpu_count()
# print('Process count:', num_process)
 


#
# EXPERIMENTAL PARAMETERS
#
D = float(100) # dilution factor
cycle = float(2) # number of cycle between data
N = float(2*10**7) # Carrying capacity: total number of cells in the flask before dilution (after growing)
epsilon = float(0.01) # initial value of epsilon, default
# Note that bottleneck size Nb = N/D

#
# BAYESIAN PARAMETERS
# initial prior of s for SModel_N is uniform distribution in range (smin, smax)
# Min s and Max s for Bayesian method (allowable \s range for adpative lineages and the value of mean fitness)
#
smax = float(2)
smin = float(-2)
NUMBER_LINEAGE_MLE = 3000# Number of lineage in likelihood function (MLE for inferring mean-fintess)
rc_default = 5
log10BF_threshold = 1.



MODEL_NAME = {'N': 'NModel', 'SN': 'SModel_N', 'SS': 'SModel_S'}
LINEAGE_TAG = {'UNK': 'Unknown', 'NEU': 'Neutral', 'ADP': 'Adaptive'}

