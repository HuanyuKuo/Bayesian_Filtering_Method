# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 11:19:49 2021

@author: huanyu
"""
from myVariables import (Constant, Global, Lineage)
from main import create_lineage_list_by_pastTag
import myConstant as mc
import myReadfile as mr

from matplotlib import pyplot as plt
import numpy as np

def read_meanfitness_BTM(lineage_info, const):
    eps_arr=[]
    sbar_arr=[]
    fdir =  mc.OutputFileDir  
    for t in range(1, const.T):
        fname = fdir + 'glob_'+ lineage_info['lineage_name'] + f'_T{t}.txt'
        f =open(fname)
        f.readline()
        f.readline()
        r = f.readline()
        eps = float( r.split('\n')[0].split('\t')[1])
        eps_arr.append(eps)
        r = f.readline()
        sbar = float( r.split('\n')[0].split('\t')[1])
        sbar_arr.append(sbar)
        f.close()
    return eps_arr, sbar_arr

def read_meanfitness_Simulation(filedirname):
    f = open(filedirname,  'r')
    line = f.readline()
    meanfitness_Simulation = []
    while(line):
        meanfitness_Simulation.append(float(line.split('\n')[0].split('\t')[1]))
        line = f.readline()
    f.close()
    return meanfitness_Simulation

def read_selection_Bayes(lins, const, lineage_info):
    
    ADP_BCID_Bayes = []
    ADP_s_mean_Bayes= []
    ADP_s_std_Bayes= []
    ADP_s_time=[]
    #lins = []
    counts = []
    #for i in range(len(bc_read)):
    #    lins.append(Lineage(bc_read[i], i))
    for t in range(2, const.T):
        lins_UNK, lins_NEU, lins_ADP = create_lineage_list_by_pastTag(lins, t, lineage_info, const)
        for lin in lins_ADP:
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
                    
    index = np.where(np.asarray(counts)>1)[0]
    ADP_BCID_Bayes = [ADP_BCID_Bayes[i] for i in index]
    ADP_s_mean_Bayes = [ADP_s_mean_Bayes[i] for i in index]
    ADP_s_std_Bayes = [ADP_s_std_Bayes[i] for i in index]
    ADP_counts = [counts[i] for i in index]
    ADP_s_time = [ADP_s_time[i] for i in index]
    return ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time
    
def read_selection_Simulation(filedirname):
    s_arr_Simulation = []
    f = open(filedirname,'r')
    line = f.readline()
    line = f.readline()
    while(line):
        # file in the order of Barcode index 0, 1, 2, .., so we store value without sorting by index
        s_arr_Simulation.append(float(line.split('\n')[0].split('\t')[1]))
        line = f.readline()
    f.close()
    return s_arr_Simulation


if __name__ == '__main__':
    
    datafilename = 'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    #datafilename = '41586_2015_BFnature14279_MOESM90_ESM_cycle_downsample'+ '.txt'  
    lins, totalread, cycles = mr.my_readfile(datafilename)
    const = Constant(totalread, cycles)
    
    case_name = 'Simulation_20222013_Population1' 
    #case_name = 'nature2015_Rep1'#'Simulation_20222013_Population1' 
    lineage_info =  {'lineage_name': case_name}
    
    
    # ##################################################
    #
    # Collect output from Bayesian Method and Save results
    #
    # ###################################################
    eps_Bayes, meanfitness_Bayes = read_meanfitness_BTM(lineage_info, const)
    ADP_BCID_Bayes, ADP_s_mean_Bayes, ADP_s_std_Bayes, ADP_counts, ADP_s_time = read_selection_Bayes(lins, const, lineage_info)
    
    
    _t_Bayes = [sum(const.Ct[1:i]) for i in range(len(const.Ct))]
    t_Bayes = [(_t_Bayes[i]+_t_Bayes[i+1])/2 for i in range(len(_t_Bayes)-1)]
    
    # Output Mean-fitness file
    f = open(mc.OutputFileDir + 'Bayesian_global_parameters_'+lineage_info['lineage_name']+'.txt','w')
    f.write('Time (cycle)\tMean-fitness(1/cycle)\tEpsilon\n')
    for i in range(len(t_Bayes)):
        f.write(str(t_Bayes[i])+'\t'+str(meanfitness_Bayes[i])+'\t'+str(eps_Bayes[i])+'\n')
    f.close()
    # Output Epsilon file
    f = open(mc.OutputFileDir + 'Bayesian_ADP_selection_coefficient_'+lineage_info['lineage_name']+'.txt','w')
    f.write('Barcode Index \t selection coefficient mean (1/cycle) \t selection coefficient std \t collect time (cycle)\t counts \n')
    for i in range(len(ADP_BCID_Bayes)):
        t_index = ADP_s_time[i]
        f.write(str(ADP_BCID_Bayes[i])+'\t'+str(ADP_s_mean_Bayes[i])+'\t'+str(ADP_s_std_Bayes[i])+'\t'+str(_t_Bayes[t_index])+ '\t'+ str(ADP_counts[i])+'\n')
    f.close()
    
    
    
    # ##################################################
    #
    # Optional Plots
    #
    # ###################################################
    FLAG_PLOT_SIMULATION = True#False
    # True: plot Bayesian inferred value with simulation value
    # Flase: plot Bayesian inferred value only
    #
    # Plot meanfitness and epsilon
    #
    if FLAG_PLOT_SIMULATION:
        simu_name = 'simuMEE_20220213'
        meanfitness_Simulation = read_meanfitness_Simulation('../simulation_MEE/'+'simulation_meanfitness_'+simu_name+ '.txt')
        t_simulation = np.arange(0,len(meanfitness_Simulation),1)
   
    
    
    
    plt.figure()
    plt.plot(t_Bayes, meanfitness_Bayes, 'bo-', label='Bayes')
    if FLAG_PLOT_SIMULATION:
        plt.plot(t_simulation, meanfitness_Simulation, 'ko-',label='Simulation')
    plt.legend()
    plt.xlabel('time (cycle)')
    plt.title('Meanfitness(1/cycle)')
    plt.savefig(mc.OutputFileDir+'meanfitness_trajectory_Bayes.png',dpi=200)
    
    plt.figure()
    plt.plot(t_Bayes, eps_Bayes, 'bo-',label='Bayes')
    if FLAG_PLOT_SIMULATION:
        plt.plot(t_simulation, [0.01 for _ in range(len(t_simulation))], 'k--', label='Simulation')
    plt.legend()
    plt.xlabel('time (cycle)')
    plt.title('Systematic Error Epsilon')
    plt.savefig(mc.OutputFileDir +'Epsilon_trajectory_Bayes.png',dpi=200)
    
    #
    # Plot DFE
    #
    bw= 0.06
    plt.figure()
    plt.hist(ADP_s_mean_Bayes, color='tab:blue', bins=int((max(ADP_s_mean_Bayes)-min(ADP_s_mean_Bayes))/bw), label='Bayes (n={:d})'.format(len(ADP_s_mean_Bayes)),alpha=0.4)
    if FLAG_PLOT_SIMULATION:
        s_Simulation = read_selection_Simulation('../simulation_MEE/' + 'simulation_selection_coefficient_'+simu_name+'.txt')
        s_Simulation = np.asarray(s_Simulation)
        ADP_BCID_Simulation = np.where(s_Simulation>0)[0]
        s_Simulation_adpative = s_Simulation[ADP_BCID_Simulation]
        
        plt.hist(s_Simulation_adpative, color='black', bins=int((max(s_Simulation_adpative)-min(s_Simulation_adpative))/bw), label='Simulation (n={:d})'.format(len(s_Simulation_adpative)),alpha=0.4)
    
    plt.legend()
    plt.xlabel('selection coefficient (1/cycle)')
    plt.title('DFE')
    plt.savefig(mc.OutputFileDir + 'DFE_Bayes.png',dpi=200)
    

    if FLAG_PLOT_SIMULATION:
        #
        # Plot selection coefficient comparision
        #
        true_positive_BCID = list(set(ADP_BCID_Simulation).intersection(set(ADP_BCID_Bayes)))
        false_positive_BCID = list(set(ADP_BCID_Bayes)-set(ADP_BCID_Simulation) )
        false_negative_BCID = list(set(ADP_BCID_Simulation)-set(ADP_BCID_Bayes) )
        
        TP_idx1 = [list(ADP_BCID_Simulation).index(bcid) for bcid in true_positive_BCID] 
        TP_idx2 = [ADP_BCID_Bayes.index(bcid) for bcid in true_positive_BCID] 
        TP_s1= s_Simulation_adpative[TP_idx1]
        TP_s2 = np.asarray(ADP_s_mean_Bayes)[TP_idx2]
        TP_s2_std = np.asarray(ADP_s_std_Bayes)[TP_idx2]*2
        plt.figure()
        plt.errorbar(x=TP_s1, y=TP_s2, yerr=TP_s2_std, fmt='o', ecolor=None,alpha=0.7,label='True positive {:d}'.format(len(TP_s2))+'/{:d}'.format(len(ADP_BCID_Bayes)))
        
        
        TP_idx2 = [ADP_BCID_Bayes.index(bcid) for bcid in false_positive_BCID] 
        TP_s2 = np.asarray(ADP_s_mean_Bayes)[TP_idx2]
        TP_s2_std = np.asarray(ADP_s_std_Bayes)[TP_idx2]*2
        x = [0 for _ in range(len(TP_s2))]
        plt.errorbar(x=x, y=TP_s2, yerr=TP_s2_std, fmt='o', ecolor=None,alpha=0.7,label='False positive {:d}'.format(len(TP_s2)))
        
        TP_idx1 = [list(ADP_BCID_Simulation).index(bcid) for bcid in false_negative_BCID] 
        TP_s1= s_Simulation_adpative[TP_idx1]
        y = [0 for _ in TP_s1]
        plt.plot(TP_s1, y, 'o',alpha=0.7, label='False negative {:d}'.format(len(TP_s1)))
        plt.title('Selection Coefficient')
        plt.xlabel('Simulation S (1/cycle)')
        plt.ylabel('Bayes S (1/cycle)')
        plt.legend()
        plt.plot([0,1.1 ],[0,1.1],'k--')
        plt.savefig(mc.OutputFileDir + 'Selection_Coefficient_Bayes.png',dpi=200)
        #
        # Plot lineage trajectory for true_positive barcode
        #
        Neff = mc.N/mc.D # the effective population size is set as the bottleneck, because  in the simulation, mutants arise at t=0 (where population size = bottleneck). 
        plt.figure()
        t_data = [const.Ct[i]*i for i in range(const.T)]
        count_suv = 0
        count_nsuv = 0
        for lin in lins:
            if lin.BCID in true_positive_BCID:
                freqs = [lin.rt[t]/totalread[t] for t in range(const.T)]
                idx1 = list(ADP_BCID_Simulation).index(lin.BCID)
                s_true = s_Simulation_adpative[idx1]
                s_true_gen = s_true / np.log2(mc.D)
                # survive means that the barcode count is "large enough" for consecutive 4 time point, such that the lineage could be identfied as ADP twiece.
                if np.sum(freqs >1/s_true_gen/Neff)>3: 
                    col = 'k'
                    count_suv+=1
                else:
                    col = 'r'
                    count_nsuv+= 1
                logfreq = np.ma.log10(freqs)
                plt.plot(t_data, logfreq,'-', color=col, lw=0.3)
        plt.xlabel('time (cycle)')
        plt.ylabel('log10 barcode freq')
        plt.title('True positive lineages ({:d} survive (black),'.format(count_suv)+' {:d} non-survive (red))'.format(count_nsuv))
        plt.savefig(mc.OutputFileDir + 'True_positive_lineages.png',dpi=200)
        #
        # Plot lineage trajectory for false_negative barcode
        #
        plt.figure()
        t_data = [const.Ct[i]*i for i in range(const.T)]
        count_suv = 0
        count_nsuv = 0
        for lin in lins:
            if lin.BCID in false_negative_BCID:
                freqs = [lin.rt[t]/totalread[t] for t in range(const.T)]
                idx1 = list(ADP_BCID_Simulation).index(lin.BCID)
                s_true = s_Simulation_adpative[idx1]
                s_true_gen = s_true / np.log2(mc.D)
                if np.sum(freqs >1/s_true_gen/Neff)>3: # survive
                    col = 'k'
                    count_suv+=1
                else:
                    col = 'r'
                    count_nsuv +=1
                logfreq = np.ma.log10(freqs)
                plt.plot(t_data, logfreq,'-', color=col, lw=0.3)
        plt.xlabel('time (cycle)')
        plt.ylabel('log10 barcode freq')
        plt.title('False negative lineages ({:d} survive (black),'.format(count_suv)+' {:d} non-survive (red))'.format(count_nsuv))
        plt.savefig(mc.OutputFileDir + 'False_negative_lineages.png',dpi=200)
    