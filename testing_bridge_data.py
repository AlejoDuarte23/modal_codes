import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
import _ssi_cov_ad_vf as SSI
import _ssi_backend as SSIb
from scipy import signal
import warnings
import sys 
import time

sys.path.insert(0,'C:/Users/aleja/Documents/metropolis_hastings/MetropolisHasting')
warnings.filterwarnings('ignore')

import _MDOF_LSQ as MDOF_LSQ
# from mcmc_engine import prior

# import tabulate
# Load .mat file
mat = loadmat('BridgeData.mat')
t, rz, wn = mat['t'], mat['rz'], mat['wn']

fs = 15 
acc = rz.T[-10000:,:]

Nc = acc.shape[1]
fo = 0.1
fi = 7.5
f = [0.19,0.31, 0.45, 0.579, 0.84,1.1]
Nm = len(f)
S = -6.5
xo = MDOF_LSQ.create_initial_list(Nm,S,f,Se = -50) 
# MDOF_LSQ.MDOF_LSQ(xo,acc,fs,fo,fi,Nm)


freq,s1,N,lenf =  MDOF_LSQ.CPSD(acc,fs,Nc,fo,fi)
# _model = MDOF_LSQ.Model(xo,freq,Nm,N)


from mcmc_engine import MCMCEngine
import warnings
from tabulate import tabulate



# model = lambda x: MDOF_LSQ.Model(x, freq, Nm, N)
def model(x):
    return MDOF_LSQ.Model(x, freq, Nm, N)

def prior(x, lim_pri):
    for i in range(len(x)):
        if not (lim_pri[i][0] <= x[i].item() <= lim_pri[i][1]):
            return 0
    return 1

def transition_model(x, std):
    return np.random.normal(x, std, len(x))

# def acceptance(x, x_new):
#     if x_new > x:
#         return True
#     else:
#         accept = 0.1
#     return accept < np.exp(x_new - x)


def print_results(results):
    table_data = []

    for i, (chain_id, result) in enumerate(results):
        accepted, _, _ = result
        means = accepted.mean(axis=0)
        row_data = [f"Chain {i + 1}"]
        row_data.extend(means)
        table_data.append(row_data)

    headers = ["Chain"]
    for i in range(len(means)):
        headers.append(f"Mean Param {i + 1}")

    print(tabulate(table_data, headers=headers, floatfmt=".4f"))

def print_results_single_run(result):
    chain_id = result[0]
    accepted = result[1][0]  # Fix the assignment here

    means = accepted.mean(axis=0)
    table_data = [[f"Chain {chain_id + 1}", means[0], means[1], means[2]]]

    headers = ["Chain", "Mean a", "Mean b", "Mean sigma"]
    print(tabulate(table_data, headers=headers, floatfmt=".4f"))


    
xo = MDOF_LSQ.create_initial_list(Nm,S,f,Se = -50) 


    

if __name__ == '__main__':

    iterations = 1000
    
    std_tr = MDOF_LSQ.generate_std(xo, Nm)
    pri_lim = MDOF_LSQ.generate_lim_pri(xo, Nm, fs)
    pri_lim = [item for sublist in pri_lim for item in sublist]


    engine = MCMCEngine(model, prior, transition_model, std_tr, pri_lim)

    num_chains =1
    initial_S_values = [-6, -7, -8, -5]

    initial_conditions = []

    for S_val in initial_S_values:
        xo = MDOF_LSQ.create_initial_list(Nm, S_val, f, Se=-20)
        initial_conditions.append(xo)
    start_time = time.time()

    result = engine.run_chain(0, initial_conditions[0], iterations, s1)
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


    # results = engine.run_chains_parallel(initial_conditions, iterations, s1, num_chains)

    # print_results(results)