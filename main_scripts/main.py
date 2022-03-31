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
from my_model_MCMCmultiprocessing import run_model_MCMCmultiprocessing, create_lineage_list_by_pastTag, add_Bayefactor_2file


MODEL_NAME = mc.MODEL_NAME
LINEAGE_TAG = mc.LINEAGE_TAG
OutputFileDir = mc.OutputFileDir
NUMBER_RAND_NEUTRAL = mc.NUMBER_LINEAGE_MLE


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
            log10bf = lin.log10_BayesFactor()
            p = min(1, log10bf)
            prob_choice.append(p)
            #prob_choice.append(0.5)
            
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
# Function randomly chooses lineages from lins
#   
def select_random_lineages_v2(lins):
    
    lins_choice =[]
    prob_choice = []
    
    for lin in lins:
        if lin.TYPETAG == LINEAGE_TAG['ADP']:
            lins_choice.append(lin)
            log10bf = lin.log10_BayesFactor()
            p = min(1, log10bf)
            prob_choice.append(p)
            
        elif lin.r0 > 0:
            lins_choice.append(lin)
            prob_choice.append(0.5)
    
    length = min(NUMBER_RAND_NEUTRAL,len(lins_choice))
    prob_choice = np.asarray(prob_choice)/sum(prob_choice)
    rand_index = np.random.choice(a=len(lins_choice), size=length, replace=False, p=prob_choice)
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
                
                add_Bayefactor_2file(lins, lineage_info['lineage_name'], current_time-1)
                # CLASSIFY LINEAGES
                lins_UNK, lins_NEU, lins_ADP = classify_lineage_by_BayesFactor(lins, current_time, lineage_info)
                
                
                # UPDATE GLOBAL VARIABLE
                # step1: Choose random lineage for liklihood function
                #lins_RAND = select_random_lineages(lins_UNK + lins_NEU+ lins_ADP)
                #lins_RAND = select_random_lineages_v2(lins_UNK + lins_NEU + lins_ADP) 
                
                if current_time == 1:
                    lins_RAND = select_random_lineages(lins_UNK + lins_NEU+ lins_ADP)
                else:
                    lins_RAND = select_small_lineages(lins_UNK + lins_NEU + lins_ADP, const.Rt[current_time-1] ) 
                
                # step2: Maximum likelihood estmiate 
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
    
    #
    # 1. FileName of Barcode Count data
    # 
    datafilename = 'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    #datafilename = '41586_2015_BFnature14279_MOESM90_ESM_cycle' + '.txt'
    #datafilename = 'Data_BarcodeCount_simuMEE_20220226' + '.txt'
    
    #
    # 2. Name of Run Case
    #
    
    #case_name = 'Simulation_20220226_NL=10e5__testRandLintoFitGlob' 
    #case_name = 'Simulation_20220226_NL=10e5__testRandLintoFitGlob' 
    #case_name = 'nature2015_E1'
    case_name = 'Simulation_20220213'
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
    # output results
    mr.output_global_parameters_BFM(lineage_info,const)
    meanfitness_Bayes_cycle, epsilon_Bayes, t_arr_cycle = mr.read_global_parameters_BFM(lineage_info)
    
    critical_BF = 2
    critical_counts = 2
    mr.output_selection_Bayes(lineage_info, datafilename, critical_BF, critical_counts)
