# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 15:15:08 2021

@author: huanyu
"""

import myConstant as mc
from myVariables import (Constant, Global, Lineage)

import numpy as np

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

if __name__ == '__main__':
    
    datafilename =  YOUR_FILE_NAME #'Data_BarcodeCount_simuMEE_20220213' + '.txt'  
    
    lins, totalread, cycles = my_readfile(datafilename)
    _const = Constant(totalread, cycles)
    for t in range(1, _const.T):
        _const.Ct[t] = cycles[t-1]
    
    
    
