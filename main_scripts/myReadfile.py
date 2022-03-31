# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:15:08 2021

@author: huanyu
"""

import myConstant as mc
from myVariables import (Constant, Global, Lineage)
from my_model_MCMCmultiprocessing import  create_lineage_list_by_pastTag
import numpy as np
from matplotlib import pyplot as plt
import os.path
#
# read input Barcode count data
def my_readfile(filename):
    
    filedirname = mc.InputFileDir + filename 
    Lins = []
    
    f = open(filedirname)
    line = f.readline()
    t_arr = line.split('\n')[0].split('\t')[1::]
    t_cycle = [ float(t.split('cycle')[0].split('=')[1]) for t in t_arr]
    cycles = [t_cycle[i+1]-t_cycle[i] for i in range(len(t_cycle)-1)]
    totalread = np.zeros(len(t_cycle))
    line = f.readline()
    while line:
        reads = line.split('\n')[0].split('\t')
        BCID = int(reads[0])
        reads = [ int(r) for r in reads[1::]]
        totalread += np.asarray(reads)
        Lins.append(Lineage(reads=reads, BCID=BCID))
        line = f.readline()
    f.close()
    return Lins, totalread, cycles

#
# output global parameters
def output_global_parameters_BFM(lineage_info, const):
    eps_Bayes = []
    meanfitness_Bayes = []
    fdir =  mc.OutputFileDir  
    for t in range(1, const.T):
        fname = fdir + 'glob_'+ lineage_info['lineage_name'] + f'_T{t}.txt'
        f =open(fname)
        f.readline()
        f.readline()
        r = f.readline()
        eps = float( r.split('\n')[0].split('\t')[1])
        eps_Bayes.append(eps)
        r = f.readline()
        sbar = float( r.split('\n')[0].split('\t')[1])
        meanfitness_Bayes.append(sbar)
        f.close()
        
    _t_Bayes = [sum(const.Ct[1:i]) for i in range(1,len(const.Ct)+1)]
    t_Bayes = [(_t_Bayes[i]+_t_Bayes[i+1])/2 for i in range(len(_t_Bayes)-1)]
    # Output Mean-fitness file
    f = open(mc.OutputFileDir + 'Bayesian_global_parameters_'+lineage_info['lineage_name']+'.txt','w')
    f.write('Time (cycle)\tMean-fitness(1/cycle)\tEpsilon\n')
    for i in range(len(t_Bayes)):
        f.write(str(t_Bayes[i])+'\t'+str(meanfitness_Bayes[i])+'\t'+str(eps_Bayes[i])+'\n')
    f.close()
    #
    # Make plots
    plt.figure()
    plt.plot(t_Bayes, meanfitness_Bayes, 'bo-', label='Bayes')
    plt.legend()
    plt.xlabel('time (cycle)')
    plt.title('Meanfitness(1/cycle)')
    plt.xlim(0, max(t_Bayes)+1)
    plt.savefig(mc.OutputFileDir+'meanfitness_trajectory_Bayes_'+lineage_info['lineage_name']+'.png',dpi=200)
    
    plt.figure()
    plt.plot(t_Bayes, eps_Bayes, 'bo-',label='Bayes')
    plt.legend()
    plt.xlabel('time (cycle)')
    plt.xlim(0, max(t_Bayes)+1)
    plt.title('Systematic Error Epsilon')
    plt.savefig(mc.OutputFileDir +'Epsilon_trajectory_Bayes_'+lineage_info['lineage_name']+'.png',dpi=200)
    
def read_global_parameters_BFM(lineage_info):
    t_arr_cycle = []
    meanfitness_Bayes_cycle = []
    epsilon_Bayes = []
    f = open(mc.OutputFileDir + 'Bayesian_global_parameters_'+lineage_info['lineage_name']+'.txt','r')
    f.readline()
    line = f.readline()
    while(line):
        line = line.split('\n')[0].split('\t')
        t_arr_cycle.append(float(line[0]))
        meanfitness_Bayes_cycle.append(float(line[1]))
        epsilon_Bayes.append(float(line[2]))
        line = f.readline()
    f.close()
    return meanfitness_Bayes_cycle, epsilon_Bayes, t_arr_cycle
'''
#
# 
def read_selection_Bayes(lineage_info, datafilename, critical_log10_BF, critical_counts):
    
    ADP_BCID_Bayes = []
    ADP_s_mean_Bayes= []
    ADP_s_std_Bayes= []
    ADP_s_time=[]
    #lins = []
    counts = []
    lins, totalread, cycles = my_readfile(datafilename)
    const = Constant(totalread, cycles)
    for t in range(1, const.T):
        const.Ct[t] = cycles[t-1]
    
    for t in range(2, const.T):
        lins = create_lineage_list_by_pastTag(lins, t, lineage_info, const)
        for lin in lins:
            if lin.TYPETAG  != mc.LINEAGE_TAG['UNK']:
                #print(lin.TYPETAG, lin.log10_BayesFactor())
                if lin.log10_BayesFactor() > critical_log10_BF:
                    
                    if lin.BCID not in ADP_BCID_Bayes:
                        ADP_BCID_Bayes.append(lin.BCID)
                        ADP_s_mean_Bayes.append(lin.sm.post_parm_NormS_mean)
                        ADP_s_std_Bayes.append(np.sqrt(lin.sm.post_parm_NormS_var))
                        ADP_s_time.append(t)
                        counts.append(1.)
                    else:
                        idx = ADP_BCID_Bayes.index(lin.BCID)
                        counts[idx] += 1.
                        if ADP_s_std_Bayes[idx] > np.sqrt(lin.sm.post_parm_NormS_var):
                            ADP_s_mean_Bayes[idx]=lin.sm.post_parm_NormS_mean
                            ADP_s_std_Bayes[idx]=np.sqrt(lin.sm.post_parm_NormS_var)
                            ADP_s_time[idx]=t
                    
    index = np.where(np.asarray(counts)>=critical_counts)[0]
    ADP_BCID_Bayes = [ADP_BCID_Bayes[i] for i in index]
    ADP_s_mean_Bayes = [ADP_s_mean_Bayes[i] for i in index]
    ADP_s_std_Bayes = [ADP_s_std_Bayes[i] for i in index]
    ADP_counts = [counts[i] for i in index]
    ADP_s_time = [ADP_s_time[i] for i in index]
    return ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time
'''

#
# output selection coefficient
def output_selection_Bayes(lineage_info, datafilename, critical_BF, critical_counts):
    
    ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time = read_selection_Bayes_v2(lineage_info, datafilename, critical_BF, critical_counts)
    
    case_name = lineage_info['lineage_name']
    f = open(mc.OutputFileDir  + 'S_Bayes_v2_'+case_name+f'_count={critical_counts}'+f'_BFthreshold={critical_BF}.txt','w')
    f.write('ADP_BCID_Bayes\tADP_s_mean_Bayes\tADP_s_std_Bayes\tADP_counts\tADP_s_time\n')
    for i in range(len(ADP_BCID_Bayes)):
        f.write(str(ADP_BCID_Bayes[i])+'\t')
        f.write(str(ADP_s_mean_Bayes[i])+'\t')
        f.write(str(ADP_s_std_Bayes[i])+'\t')
        f.write(str(ADP_counts[i])+'\t')
        f.write(str(ADP_s_time[i])+'\n')
    f.close()

def get_tmp_s_from_BayesFiles(lineage_info, t):
    
    MODEL_NAME = mc.MODEL_NAME
    OutputFileDir = mc.OutputFileDir
    lineage_name = lineage_info['lineage_name']
    
    tmp_bcids_SS = []
    tmp_s_mean_SS = []
    tmp_s_var_SS = []
    tmp_log10bf_SS = []
    readfilename = 'posterior_'+lineage_name+'_'+MODEL_NAME['SS']+f"_T{t}.txt"
    print(readfilename)
    if os.path.exists(OutputFileDir +readfilename) is False:
        print('No file '+OutputFileDir +readfilename+'\n')
    else:
        f = open(OutputFileDir +readfilename ,'r')
        a = f.readlines()
        f.close()
        for i in range(1, len(a)):
            b = a[i].split('\n')[0].split('\t')
            if  (float(b[6]) >0) and (len(b)>10) :
                tmp_bcids_SS.append(  int(b[1]))
                tmp_s_mean_SS.append( float(b[6]))
                tmp_s_var_SS.append( float(b[7]))
                tmp_log10bf_SS.append( float(b[10]))
    tmp_bcids_SN = []
    tmp_s_mean_SN = []
    tmp_s_var_SN = []
    tmp_log10bf_SN = []
    readfilename = 'posterior_'+lineage_name+'_'+MODEL_NAME['SN']+f"_T{t}.txt"
    print(readfilename)
    if os.path.exists(OutputFileDir +readfilename) is False:
        print('No file '+OutputFileDir +readfilename+'\n')
    else:
        f = open(OutputFileDir +readfilename ,'r')
        a = f.readlines()
        f.close()
        for i in range(1, len(a)):
            b = a[i].split('\n')[0].split('\t')
            if  (float(b[6]) >0) and (len(b)>10) :
                tmp_bcids_SN.append(  int(b[1]))
                tmp_s_mean_SN.append( float(b[6]))
                tmp_s_var_SN.append( float(b[7]))
                tmp_log10bf_SN.append( float(b[10]))
    return tmp_bcids_SS, tmp_s_mean_SS, tmp_s_var_SS, tmp_log10bf_SS, tmp_bcids_SN, tmp_s_mean_SN, tmp_s_var_SN, tmp_log10bf_SN

def read_selection_Bayes_v2(lineage_info, datafilename, critical_BF, critical_counts):
    
    ADP_BCID_Bayes = []
    ADP_s_mean_Bayes= []
    ADP_s_std_Bayes= []
    ADP_s_time=[]
    #lins = []
    counts = []
    
    critical_log10_BF = np.log10(critical_BF)
    
    lins, totalread, cycles = my_readfile(datafilename)
    
    total_timepoint = len(totalread)
    
    tend = total_timepoint -1 
    t = tend
    
    MODEL_NAME = mc.MODEL_NAME
    OutputFileDir = mc.OutputFileDir
    lineage_name = lineage_info['lineage_name']
    
    tmp_bcids_SS = []
    tmp_s_mean_SS = []
    tmp_s_var_SS = []
    readfilename = 'posterior_'+lineage_name+'_'+MODEL_NAME['SS']+f"_T{t}.txt"
    if os.path.exists(OutputFileDir +readfilename) is False:
        print('No file '+OutputFileDir +readfilename+'\n')
    else:
        f = open(OutputFileDir +readfilename ,'r')
        a = f.readlines()
        f.close()
        for i in range(1, len(a)):
            b = a[i].split('\n')[0].split('\t')
            if  float(b[6]) >0:
                tmp_bcids_SS.append(  int(b[1]))
                tmp_s_mean_SS.append( float(b[6]))
                tmp_s_var_SS.append( float(b[7]))
                
    for i in range(len(tmp_bcids_SS)):
        if tmp_bcids_SS[i] > 0:
            bcid = tmp_bcids_SS[i]
            s_mean = tmp_s_mean_SS[i]
            s_std = np.sqrt(tmp_s_var_SS[i])
            
            ADP_BCID_Bayes.append(bcid)
            ADP_s_mean_Bayes.append(s_mean)
            ADP_s_std_Bayes.append(s_std)
            ADP_s_time.append(t)
            counts.append(1.)
        
    t_arr = [t-i for i in range(1,t-1)]
    prev_bcids = [bcid for bcid in tmp_bcids_SS]
    
    for t in t_arr:
        print(t)
        tmp_bcids_SS, tmp_s_mean_SS, tmp_s_var_SS, tmp_log10bf_SS, tmp_bcids_SN, tmp_s_mean_SN, tmp_s_var_SN, tmp_log10bf_SN = get_tmp_s_from_BayesFiles(lineage_info, t)
        
        bcid_dict_SS = {}
        for i in range(len(tmp_bcids_SS)):
            bcid_dict_SS.update({tmp_bcids_SS[i]:i})
        bcid_dict_SN = {}
        for i in range(len(tmp_bcids_SN)):
            bcid_dict_SN.update({tmp_bcids_SN[i]:i})
        #print(bcid_dict_SS.keys())
        for j in range(len(prev_bcids)):
            bcid = prev_bcids[j]
            #print(t,len(prev_bcids),j, bcid)
            if bcid in bcid_dict_SS.keys():
                i = bcid_dict_SS[bcid]#np.where(np.asarray(tmp_bcids_SS))[0][0]
                bcid = tmp_bcids_SS[i]
                s_mean = tmp_s_mean_SS[i]
                s_std = np.sqrt(tmp_s_var_SS[i])
                log10bf = tmp_log10bf_SS[i]
            elif bcid  in bcid_dict_SN.keys():
                i = bcid_dict_SN[bcid]#i = np.where(np.asarray(tmp_bcids_SN))[0][0]
                bcid = tmp_bcids_SN[i]
                s_mean = tmp_s_mean_SN[i]
                s_std = np.sqrt(tmp_s_var_SN[i])
                log10bf = tmp_log10bf_SN[i]
            else:
                print('Warning! ')
                log10bf =''
                
            if log10bf >= critical_log10_BF:
                if bcid not in ADP_BCID_Bayes:
                    ADP_BCID_Bayes.append(bcid)
                    ADP_s_mean_Bayes.append(s_mean)
                    ADP_s_std_Bayes.append(s_std)
                    ADP_s_time.append(t)
                    counts.append(1.)
                else:
                    idx = ADP_BCID_Bayes.index(bcid)
                    counts[idx] += 1.
                    if ADP_s_std_Bayes[idx] > s_std:
                        ADP_s_mean_Bayes[idx]=s_mean
                        ADP_s_std_Bayes[idx]=s_std
                        ADP_s_time[idx]=t
        prev_bcids = [bcid for bcid in tmp_bcids_SS]
    
    index = np.where(np.asarray(counts)>=critical_counts)[0]
    ADP_BCID_Bayes = [ADP_BCID_Bayes[i] for i in index]
    ADP_s_mean_Bayes = [ADP_s_mean_Bayes[i] for i in index]
    ADP_s_std_Bayes = [ADP_s_std_Bayes[i] for i in index]
    ADP_counts = [counts[i] for i in index]
    ADP_s_time = [ADP_s_time[i] for i in index]
    
    return ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time
    
if __name__ == '__main__':
    
    datafilename =  'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    
    lins, totalread, cycles = my_readfile(datafilename)
    _const = Constant(totalread, cycles)
    for t in range(1, _const.T):
        _const.Ct[t] = cycles[t-1]
    
    
    
