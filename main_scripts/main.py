# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 09:50:31 2020

@author: huanyu
"""

#from matplotlib import pyplot as plt
import numpy as np
#import scipy
#import time
#import os.path

import myConstant as mc
import myReadfile as mr
from myVariables import (Constant, Global, Lineage)
from my_model_MCMCmultiprocessing import run_model_MCMCmultiprocessing, readfile2lineage, add_Bayefactor_2file


MODEL_NAME = mc.MODEL_NAME
LINEAGE_TAG = mc.LINEAGE_TAG
OutputFileDir = mc.OutputFileDir
NUMBER_RAND_NEUTRAL = mc.NUMBER_LINEAGE_MLE


#
# Function creates the tag list of lineage from past time point
#   
def create_lineage_list_by_pastTag(lins, current_time, lineage_info, const):
    
    last_time = current_time -1
    
    # Update the reads value to current time
    for lin in lins:
        lin.set_reads(last_time=last_time)
        
    #
    # Read The PAST infromation from file and read PastTAG of lineage
    #
    if last_time ==0:
        for lin in lins:
            mu_r = float((0.001+lin.r0))
            k = mu_r/(1+mu_r*const.eps)
            theta = (1+mu_r*const.eps)/const.Rt[0]*const.Nt[0]
            #lin.nm.UPDATE_POST_PARM(k=lin.r0+0.001, theta=float(const.Nt[0]/const.Rt[0]),  log_norm= 0., log_prob_survive=0.)
            lin.nm.UPDATE_POST_PARM(k=k, theta=theta, log_norm= 0., log_prob_survive=0.)
            
            lin._init_TAG()
            
    elif last_time >0:
        lins_survive = []
        for lin in lins:
            if lin.T_END > current_time:
                lins_survive.append(lin)
                
        lins = lins_survive
        lins = readfile2lineage(lins, lineage_info['lineage_name'], last_time)
        
        add_Bayefactor_2file(lins, lineage_info['lineage_name'], last_time)
    
    '''
    # Note that the tag is still the past tag
    lins_UNK = []
    lins_NEU = []
    lins_ADP = []
    
    for i in range(len(lins)):
        
        typetag = lins[i].TYPETAG
        
        if typetag == LINEAGE_TAG['UNK']:
            lins_UNK.append(lins[i])
        elif typetag == LINEAGE_TAG['NEU']:
            lins_NEU.append(lins[i])
        elif typetag == LINEAGE_TAG['ADP']:
            lins_ADP.append(lins[i])
    
    if last_time >0:
        # Based on Tag, read SModel from different file
        lins_NEU = readfile2lineage(lins_NEU, lineage_info['lineage_name'], MODEL_NAME['SN'], last_time)
        lins_ADP = readfile2lineage(lins_ADP, lineage_info['lineage_name'], MODEL_NAME['SS'], last_time)
        
    lins 
        # Update Bayes Factor information for NEU and ADP group
        add_Bayefactor_2file(lins_UNK, lineage_info['lineage_name'], MODEL_NAME['SN'], last_time)
        add_Bayefactor_2file(lins_ADP, lineage_info['lineage_name'], MODEL_NAME['SS'], last_time)
    '''
    return lins

#
# Function randomly chooses "small neutral lineages" and "adpative lineages" from lins
#   
def select_small_lineages(lins, R):
    
    rc = max(10,0.0001*R)#max(5, min(30, 30* 10**(np.log10(R)-5)))
    print('rc=',rc)
    #rc = 30
    lins_choice =[]
    prob_choice = []
    
    for lin in lins:
        if lin.TYPETAG == LINEAGE_TAG['ADP']:
            lins_choice.append(lin)
            prob_choice.append(0.5)
            
        elif (lin.r0 > 0) and (lin.r0 < rc):
            lins_choice.append(lin)
            prob_choice.append(1-lin.r0/rc)
    
    length = min(NUMBER_RAND_NEUTRAL,len(lins_choice))
    prob_choice = np.asarray(prob_choice)/sum(prob_choice)
    rand_index = np.random.choice(a=len(lins_choice), size=length, replace=False, p=prob_choice)
    lins_ = [ lins_choice[i] for i in list(rand_index)]
    
    return lins_

#
# Function randomly chooses lineages from lins
#   
def select_random_lineages(lins):
    
    lins_choice =[]
    
    for lin in lins:
        if lin.r0 > 0:
            lins_choice.append(lin)
    
    length = min(NUMBER_RAND_NEUTRAL,len(lins_choice))
    rand_index = np.random.choice(a=len(lins_choice), size=length, replace=False)
    lins_ = [ lins_choice[i] for i in list(rand_index)]
    
    return lins_
#
# Function classifies lineage to different list based on the TAG
#   
def classify_lineage_by_BayesFactor(lins, current_time, lineage_info):
    
    #rc = Ratio*glob.epsilon
    last_time = current_time - 1
    
    # Note that the tag is new tag
    lins_UNK = []
    lins_NEU = []
    lins_ADP = []
    
    for i in range(len(lins)):
        
        lins[i].reTAG(last_time=last_time)
        typetag = lins[i].TYPETAG
        
        if typetag == LINEAGE_TAG['UNK']:
            lins_UNK.append(lins[i])
        elif typetag == LINEAGE_TAG['NEU']:
            lins_NEU.append(lins[i])
        elif typetag == LINEAGE_TAG['ADP']:
            lins_ADP.append(lins[i])
        #elif typetag == LINEAGE_TAG['EXT']:
        #    f.write(str(lins[i].BCID)+'\t'+str(last_time)+'\n')
    #f.close()
    return lins_UNK, lins_NEU, lins_ADP
  
def run_lineages(lins, start_time, end_time, const, lineage_info):
    #s_bar = []
    if (end_time <= const.T) and (start_time >=0 ):    
        glob = Global(const)    
        
        for current_time in range(start_time, end_time):
            
            if current_time >0 :
                
                # READ LINEAGE FROM THE PAST FILES
                lins = create_lineage_list_by_pastTag(lins, current_time, lineage_info, const)
                
                # CLASSIFY LINEAGES
                lins_UNK, lins_NEU, lins_ADP = classify_lineage_by_BayesFactor(lins, current_time, lineage_info)
                
                
                
                # UPDATE GLOBAL VARIABLE
                # step1: Choose random lineage for liklihood function
                if current_time == 1:
                    lins_RAND = select_random_lineages(lins_UNK + lins_NEU+ lins_ADP)
                else:
                    lins_RAND = select_small_lineages(lins_UNK + lins_NEU + lins_ADP, const.Rt[current_time-1] ) 
                # step2: Maximum likelihood estmiate 
                if current_time <= 3:
                    glob.UPDATE_GLOBAL(current_time, const, lineage_info, lins_RAND, '2d') # 2d: optimize meanfitness and epsilon
                else:
                    #glob.UPDATE_GLOBAL(current_time, const, lineage_info, lins_RAND, '1d') # 1d: optimize meanfitness
                    glob.UPDATE_GLOBAL(current_time, const, lineage_info, lins_RAND, '2d')
                
                
                # COMPUTE POSTERIOR BY MULTIPROCESSING MCMC
                # run NModel for all lineages
                #
                if len(lins_UNK + lins_NEU + lins_ADP)>0:
                    run_dict = {'model_name': MODEL_NAME['N'], 'lineage_name': lineage_info['lineage_name']}
                    run_model_MCMCmultiprocessing(run_dict, lins_UNK + lins_NEU + lins_ADP, glob)  
               
                # run SModel_N for NEU lineages
                #
                if len(lins_NEU)>0:
                    run_dict = {'model_name': MODEL_NAME['SN'], 'lineage_name': lineage_info['lineage_name']}
                    run_model_MCMCmultiprocessing(run_dict, lins_NEU, glob) 
                
                # run SModel_S for ADP lineages
                #
                if len(lins_ADP)>0:
                    run_dict = {'model_name': MODEL_NAME['SS'], 'lineage_name': lineage_info['lineage_name']}
                    run_model_MCMCmultiprocessing(run_dict, lins_ADP, glob) 
                
    else:
        print(f"the input start_time ={start_time} must >=0 & the end_time ={end_time} must <= total time point {const.T}")
    #print(s_bar)
    #return lins_RAND, glob, s_bar#lins

if __name__ == '__main__':
    
    # ##################################
    # Set your filename and case_name
    # ################################## 
    
    # FileName of Barcode Count data
    datafilename = 'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    #datafilename = '41586_2015_BFnature14279_MOESM90_ESM_cycle' + '.txt'
    # Name of Run Case
    case_name = 'Simulation_20222013_Population1' 
    #case_name = 'nature2015_E1'
    
    #
    # create lineage & constant
    lins, totalread, cycles = mr.my_readfile(datafilename)
    const = Constant(totalread, cycles)
    for t in range(1, const.T):
        const.Ct[t] = cycles[t-1]
    #
    # run Bayesian Filtering Method
    start_time = 1
    end_time = const.T
    lineage_info =  {'lineage_name': case_name}
    run_lineages(lins, start_time, end_time, const, lineage_info)
    
    #
    # output result
    
    