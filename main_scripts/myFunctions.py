# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:07:41 2020

@author: huanyu
"""
import numpy as np
import scipy
import scipy.stats
from scipy import optimize
import myConstant as mc
LINEAGE_TAG = mc.LINEAGE_TAG#= {'UNK': 'Unknown', 'NEU': 'Neutral', 'ADP': 'Adaptive'}
        
        
def posterior_constant_neutral(transition, epsilon, k, theta, read, D):
    #
    # transition = RD/N * exp(-bar_s * C),
    # where R = total read, D=dilution ratio, N=population size, bar_s = mean fitness per cycle, C = cycle legth 
    # 
    NUM_SAMPLE = 10000
    alpha_arr = np.random.negative_binomial(n=k, p=1/(1+theta/D), size=NUM_SAMPLE)
    n_arr_parameter_observation_NB = alpha_arr * transition / epsilon
    #print(n_arr_parameter_observation_NB)
    p_parameter_observation_NB = 1/(1+epsilon)
    prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_arr_parameter_observation_NB,p=p_parameter_observation_NB)
    prob_observation_NB = np.ma.masked_invalid(prob_observation_NB)
    prob_observation_NB = prob_observation_NB.filled(0)
    expected_prob_observation = np.mean(prob_observation_NB)
    #expected_prob_observation = expected_prob_observation.filled(0)
    return expected_prob_observation

def posterior_constant_selective(transition, epsilon, k, a, b, mean_s, var_s, read, D, C):
    #
    # transition = RD/N * exp(-bar_s * C),
    # where R = total read, D=dilution ratio, N=population size, bar_s = mean fitness per cycle, C = cycle legth 
    #
    
    
    NUM_SAMPLE = 100000
    #
    # x = ( s - mean_s )/sigma_s
    # sigma_s = standard deviation of selection s for P(s) = prior dist
    # 
    x_arr = np.random.normal(size=NUM_SAMPLE)
    theta_arr = a/k * np.exp(b*x_arr)
    #print(theta_arr)
    p_arr_parameter_analytical_Prior = 1/(1+theta_arr/D)
    alpha_arr = np.random.negative_binomial(n=k, p=p_arr_parameter_analytical_Prior)
    #print(alpha_arr)
    
    s_arr = x_arr*(var_s**.5) + mean_s
    transition_arr = transition * np.exp(s_arr*C)
    
    n_arr_parameter_observation_NB = alpha_arr * transition_arr / epsilon
    p_parameter_observation_NB = 1/(1+epsilon)
    #print('n',n_arr_parameter_observation_NB)
    #print('p',p_parameter_observation_NB)
    #print('expect',n_arr_parameter_observation_NB*(1-p_parameter_observation_NB)/p_parameter_observation_NB)
    prob_observation_NB = scipy.stats.nbinom.pmf(k=read,n=n_arr_parameter_observation_NB,p=p_parameter_observation_NB)
    #print(prob_observation_NB)
    prob_observation_NB = np.ma.masked_invalid(prob_observation_NB)
    prob_observation_NB = prob_observation_NB.filled(0)

    #prob_selection = scipy.stats.norm.pdf(x_arr)
    
    #prob_2d = prob_selection* prob_observation_NB
    
    expected_prob_observation = np.mean(prob_observation_NB)
    
    return expected_prob_observation
   
def likelihood_function_global_variable(meanfitness, epsilon, glob, lins):
    lins_ADP = []
    lins_NEU = []
    
    llk = 0
    
    D = glob.D
    C = glob.C
    transition = np.exp(-1.*meanfitness * C+ np.log(D) + np.log(glob.R) - np.log(glob.N) )
    
    for i in range(len(lins)):
        if lins[i].TYPETAG == LINEAGE_TAG['ADP']:
            lins_ADP.append(lins[i])
        elif lins[i].TYPETAG == LINEAGE_TAG['NEU']:
            lins_NEU.append(lins[i])
    
    for lin in lins_ADP:
        k = lin.sm.post_parm_Gamma_k
        a = lin.sm.post_parm_Gamma_a
        b = lin.sm.post_parm_Gamma_b
        mean_s = lin.sm.post_parm_NormS_mean
        var_s = lin.sm.post_parm_NormS_var
        read = lin.r1
        p = posterior_constant_selective(transition, epsilon, k, a, b, mean_s, var_s, read, D, C)
        llk += p
        
    for lin in lins_NEU:
        k = lin.nm.post_parm_Gamma_k
        theta = lin.nm.post_parm_Gamma_theta
        read = lin.r1
        p=posterior_constant_neutral(transition, epsilon, k, theta, read, D)
        llk += p
    
    llk = np.log(llk) - np.log(len(lins))
    return llk

def Maximum_Likelihood_globals_twostep(glob, lins, optimize_method, const, t):
    alphas=[]
    reads = []
    r0=0
    r1=0
    for i in range(len(lins)):
        if lins[i].TYPETAG == LINEAGE_TAG['ADP']:
            r0 = r0 + lins[i].r0*np.exp(lins[i].sm.post_parm_NormS_mean * glob.C)
            r1 = r1 + lins[i].r1
            reads.append(lins[i].r1)
            alphas.append( np.exp( lins[i].sm.post_parm_NormS_mean*glob.C + np.log(lins[i].sm.post_parm_Gamma_a)+ 0.5*(lins[i].sm.post_parm_Gamma_b)**2 -np.log(glob.D)))
        elif (lins[i].nm.post_parm_Gamma_k>1):
            r0 = r0 + lins[i].r0
            r1 = r1 + lins[i].r1
            reads.append(lins[i].r1)
            alphas.append( (lins[i].nm.post_parm_Gamma_k-1)* np.exp( np.log(lins[i].nm.post_parm_Gamma_theta) -np.log(glob.D)))

    s_bar_from_read = (np.log(r0)-np.log(r1) - np.log(const.Rt[t-1]) + np.log(const.Rt[t]))/glob.C
    alphas = np.asarray(alphas)
    reads = np.asarray(reads)
    
    
    # Step 1
    # MLE for meanfitness estimates
    
    method = optimize_method
    transition = np.exp(-1.*s_bar_from_read*glob.C)#r1/r0# initial value
    sol=[]
    
    bounds = optimize.Bounds([0.], [np.inf])
    sol = optimize.minimize(fun= new_llk_trans, x0 = np.array([transition]), args=(alphas, reads, glob), method=method,bounds=bounds)
    if sol.success == True:
        transition = sol.x[0]
    else:
        print("\n Warning: MLE fails to find optimal value for meanfitness")
        print(alphas)
    #print(sol)
    
    
    
    # Step 2
    # MLE for epsilon estimates
    
    epsilon = 10. # initial value
    sol2=[]
    bounds = optimize.Bounds([0.1], [20])
    expected_reads = np.asarray(alphas)* transition* np.exp(np.log(glob.D) + np.log(glob.R) - np.log(glob.N))
    sol2 = optimize.minimize(fun= new_llk_epsilon, x0 = np.array([epsilon]), args=(reads, expected_reads), method=method,bounds=bounds)
    if sol2.success == True:
        epsilon = sol2.x[0]
    else:
        print("\n Warning: MLE fails to find optimal value for epsilon")
    #print(sol2)
    
    meanfitness_est = -1*np.log(transition)/glob.C#max(-1*np.log(transition)/glob.C, 0)
    epsilon_est = epsilon
    
    return meanfitness_est, epsilon_est, sol, sol2, s_bar_from_read

# return the likelihood function of parameter- transition
def new_llk_trans(transition, alphas, reads, glob):
    result = 0
    exprexted_reads = alphas *transition* np.exp(np.log(glob.D) + np.log(glob.R) - np.log(glob.N))
    for i in range(len(reads)):
        result = result + -1.*exprexted_reads[i] + reads[i]*np.log(exprexted_reads[i]) - scipy.special.gammaln(1+reads[i])
    return -1.* result

def new_llk_epsilon(epsilon, reads, expected_reads):
    result = 0
    p = 1./(1.+epsilon)
    n_arr = expected_reads/ epsilon
    for i in range(len(reads)):
        result = result + scipy.stats.nbinom.logpmf(k=reads[i], n=n_arr[i], p=p)
        
    return -1.*result

def analytical_Posterior_cellnum_SModel(x_arr, k, a, b, data_s):
    mean_s = np.mean(data_s)
    std_s = np.std(data_s)
    ds_arr = b*(data_s-mean_s)/std_s
    p_gamma = []
    for x in x_arr:
        x2theta_arr =[ x/a*k / np.exp(ds) for ds in ds_arr]
        ave = np.mean([(x2th**k) * np.exp(-x2th) for x2th in x2theta_arr ])
        p_gamma.append( ave/x/scipy.special.gamma(k))
    return np.asarray(p_gamma)

def analytical_Posterior_log10cellnum_SModel(x_arr, k, a, b, data_s):
    mean_s = np.mean(data_s)
    std_s = np.std(data_s)
    ds_arr = b*(data_s-mean_s)/std_s
    p_gamma = []
    x_arr = np.exp(x_arr*np.log(10))
    
    for x in x_arr:
        x2theta_arr =[ x/a*k / np.exp(ds) for ds in ds_arr]
        ave = np.mean([(x2th**k) * np.exp(-x2th) for x2th in x2theta_arr ])
        p_gamma.append( ave/scipy.special.gamma(k)*np.log(10) )
    
    '''
    k2 = k/((1+k)*np.exp(b*b)-k)
    theta2 = a/k2 * np.exp(b*b/2)
    for x in x_arr:
        x2theta2 = x/theta2
        tmp = (x2theta2**k2) * np.exp(-x2theta2)
        p_gamma.append(tmp/scipy.special.gamma(k2)*np.log(10))
    '''    
    return np.asarray(p_gamma)

def get_k2_theta2(k,a,b):
    _k2 = k2(k,a,b)
    _theta2 = theta2(_k2, a, b)
    return _k2, _theta2

def k2(k,a,b):
    return k/((1+k)*np.exp(b*b)-k)

def theta2(k2,a,b):
    return a/k2 * np.exp(b*b/2)

class S_model_posterior():
    def __init__(self, data_cell_num, data_selection_coefficient, log_joint_prob):
        # data = MC sampling of cellnumber, selection coefficient
        data_cellnum = np.asarray(data_cell_num)
        data_s = np.asarray(data_selection_coefficient)
        self.data_logcellnum = np.log(data_cellnum)
        
        self.var_s = np.var(data_s)
        self.mean_s = np.mean(data_s)
        self.delta_s = (data_s- self.mean_s)/ np.std(data_s)
        self.mean_data_logcellnum = np.mean(self.data_logcellnum)
        
        self.b0 = np.cov(self.data_logcellnum, data_s)[1,0]/np.std(data_s)
        self.a0 = np.mean(data_cellnum) * np.exp(-1.*self.b0 *self.b0 /2)
        sol = optimize.root_scalar(f = self._find_k0, bracket=[0, 1000000], method='brentq')
        #self.k0 = max(sol.root, 1)
        self.k0 = sol.root
        self.k = self.k0
        self.a = self.a0
        self.b = self.b0
        
        logp_post = self._log_posterior(data_cell_num, data_selection_coefficient)
        self.log_normalization_const = self._get_log_normalization_const(logp_post, log_joint_prob)
        
    def _find_k0(self, x):
        return scipy.special.polygamma(1,x) - self.var_s
    
    def _fn_llk_neg(self, x):
        k = x[0]
        a = x[1]
        b = x[2]
        arr = self.data_logcellnum - b * self.delta_s
        y0 = k * self.mean_data_logcellnum
        y1 = -1.*k*np.mean(np.exp(arr))/a
        y2 = k*(np.log(k)-np.log(a)) - scipy.special.gammaln(k)
        y = y0+y1+y2
        
        return -1.*y
    
    def maximum_llk_S_Model_GammaDist_Parameters(self): 
        # Maximum Likelihood Estimates MLE: estimates of parameters (k, a, b) of gamm distribuiton
        
        # bounds of parmater: 0 < k < inf, 0 < a < inf, 0 < b < inf
        bounds = optimize.Bounds([0.0, 1.0, 0.0], [np.inf, np.inf, np.inf])
        method = 'L-BFGS-B'#'TNC' #'L-BFGS-B'
        sol = optimize.minimize(fun= self._fn_llk_neg, x0 = np.array([self.k0, self.a0, self.b0]), method=method, bounds=bounds)
        if sol.success == True:
            self.k = sol.x[0]
            self.a = sol.x[1]
            self.b = sol.x[2]
        else:
            print("\n Warning: MLE fails to find optimal value for parameters (k, theta) of gamma distribuiton.")

        return sol
    
    def _get_log_normalization_const(self, log_posterior, log_joint_prob):
        arr = log_joint_prob-log_posterior
        return np.median(arr)
    
    def _log_posterior(self, data_cellnum, data_s):
        # return log of posterior P(s, n) = log P(s) + log P(n|s)
        std_s = np.std(data_s)
        n2th  = [np.exp( np.log(self.k) + np.log(n) - np.log(self.a) - self.b*(s-self.mean_s)/std_s) for n,s in zip(data_cellnum, data_s)] 
        logps = [-1.*(s-self.mean_s)*(s-self.mean_s)/(2*self.var_s)  for s in data_s] 
        logps = logps + -1.*np.log(2*np.pi*std_s*std_s)/2.
        logpn = [self.k*np.log(x) - x - np.log(n) for n,x in zip(data_cellnum, n2th)] 
        logpn = logpn - scipy.special.gammaln(self.k)
        logp  = logps + logpn
        return logp
    
    
class N_model_posterior():
    def __init__(self, data, log_joint_prob):
        # data = MC sampling of cellnumber
        # log_joint_prob = log of Joint probability (Prior X Likelihood)
        
        self._mean_cellnumber = np.mean(data)
        self._var_cellnumber = np.var(data)
        self._mean_logcellnumber = np.mean(np.log(data))
        
        self.theta0 = self._var_cellnumber/self._mean_cellnumber
        self.k0 = self._mean_cellnumber/self.theta0 #max(self._mean_cellnumber/self.theta0, 1)
        
        self.k = self.k0
        self.theta = self.theta0
        
        self.sol = self._maximum_llk_N_Model_GammaDist_Parameters()
        
        logp_post = self._log_posterior_cellnumber(data)
        self.log_normalization_const = self._get_log_normalization_const(logp_post, log_joint_prob)
    
    def _log_gamma_dist(self, x, k, theta):
        # return log pdf of gamma distribution of x with parameters shape = k, scale = theta
        logp = (k-1.)*np.log(x) - np.exp(np.log(x)-np.log(theta)) - scipy.special.gammaln(k) - k*np.log(theta)
        return logp

    def _fn_llk_neg(self, x):
        # negative likelihood function of fitting parameter (k, theta) of gamma distribution 
        k = x[0]
        theta = x[1]
        y =  (k-1.)*self._mean_logcellnumber - np.exp(np.log(self._mean_cellnumber)-np.log(theta)) - scipy.special.gammaln(k) - k*np.log(theta)
        return -1.*y

    def _maximum_llk_N_Model_GammaDist_Parameters(self):
        # Maximum Likelihood Estimates MLE: estimates of parameters (k, theta) of gamm distribuiton

        # bounds of parmater: 1 < k < inf, and 0 < theta < inf
        bounds = optimize.Bounds([0.0, 0.00001], [np.inf, np.inf])
        method = 'L-BFGS-B'#'TNC' #'L-BFGS-B'
        sol = optimize.minimize(fun= self._fn_llk_neg, x0 = np.array([ self.k0, self.theta0]), method=method, bounds=bounds)
        if sol.success == True:
            self.k = sol.x[0]
            self.theta = sol.x[1]
        else:
            print("\n Warning: MLE fails to find optimal value for parameters (k, theta) of gamma distribuiton.")
        return sol
            
    def _log_posterior_cellnumber(self, y):
        # return log of P(y) is gamma distribution
        k = self.k
        theta = self.theta 
        #y = np.exp(x)
        logpy = self._log_gamma_dist(y, k, theta)
        return logpy
    
    def _get_log_normalization_const(self, log_posterior, log_joint_prob):
        arr = log_joint_prob-log_posterior
        return np.median(arr) 