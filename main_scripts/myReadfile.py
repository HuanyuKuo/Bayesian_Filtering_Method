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
if __name__ == '__main__':
    
    datafilename =  'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    
    lins, totalread, cycles = my_readfile(datafilename)
    _const = Constant(totalread, cycles)
    for t in range(1, _const.T):
        _const.Ct[t] = cycles[t-1]
    
    
    
