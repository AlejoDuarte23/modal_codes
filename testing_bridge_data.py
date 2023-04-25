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
    return MDOF_LSQ.Model_opt(x, freq, Nm, N)



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
    accepted_values = result[1][0]
    num_params = accepted_values.shape[1]

    means = accepted_values.mean(axis=0)

    table_data = [["Chain"] + [f"Mean param {i + 1}" for i in range(num_params)]]
    table_data.append([f"Chain {chain_id + 1}"] + list(means))

    print(tabulate(table_data, headers="firstrow", floatfmt=".4f"))

def print_results_multiple_chains(results):
    if len(results) == 0:
        print("No results found.")
        return
    
    num_params = results[0][1][0].shape[1] if results[0][1][0].size != 0 else results[0][1][1].shape[1]

    table_data = [["Chain"] + [f"Mean param {i + 1}" for i in range(num_params)]]

    for result in results:
        chain_id = result[0]
        accepted_values = result[1][0]
        
        if accepted_values.size == 0:
            means = [float('nan') for _ in range(num_params)]
        else:
            means = accepted_values.mean(axis=0)
            means = [round(mean, 4) for mean in means]

        table_data.append([f"Chain {chain_id + 1}"] + list(means))

    # Print the table
    for row in table_data:
        print(" | ".join([str(x).ljust(12) for x in row]))

import matplotlib.pyplot as plt



def plot_histograms_per_chain(results):
    if len(results) == 0:
        print("No results found.")
        return

    Nm = results[0][1][0].shape[1] // 2 if results[0][1][0].size != 0 else results[0][1][1].shape[1] // 2
    num_params = 2 * Nm

    for result in results:
        chain_id = result[0]
        accepted_values = result[1][0]

        if accepted_values.size == 0:
            continue

        fig, axes = plt.subplots(1, num_params, figsize=(num_params * 4, 4))
        fig.suptitle(f"Chain {chain_id + 1}")

        for param_index in range(num_params):
            ax = axes[param_index]
            ax.hist(accepted_values[:, param_index], bins=30, density=True, alpha=0.7)
            ax.set_title(f"Param {param_index + 1}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()






def plot_matrix(results, chain_ids, Nm):
    num_chains = len(chain_ids)
    num_params = 2 * Nm
    fig, axes = plt.subplots(num_params, num_params, figsize=(num_params * 3, num_params * 3))

    colors = ['black']

    for k in range(num_chains):
        chain_id = chain_ids[k]
        accepted_values = results[chain_id][1][0]

        for i in range(num_params):
            for j in range(num_params):
                ax = axes[i, j]
                ax.set_facecolor('white')

                if i == j:
                    # Histogram on the diagonal
                    ax.hist(accepted_values[:, i], bins=30, color=colors[k % len(colors)], alpha=0.7, histtype='step', label=f"Chain {chain_id + 1}")
                    ax.set_title(f"Param {i + 1}")
                else:
                    # Hexbin plot off the diagonal
                    hb = ax.hexbin(accepted_values[:, j], accepted_values[:, i], gridsize=50, cmap='gray_r', alpha=0.7)

                if i < num_params - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()

# Call the function with the `results`, a list of `chain_ids`, and `Nm`
chain_ids = [2]  # Replace with the chain ids you want to include
Nm = 7  # Replace with the value of Nm
# plot_matrix(results, chain_ids, Nm)










    
xo = MDOF_LSQ.create_initial_list(Nm,S,f,Se = -50) 


    

if __name__ == '__main__':

    iterations = 10000
    
    std_tr = MDOF_LSQ.generate_std(xo, Nm)
    pri_lim = MDOF_LSQ.generate_lim_pri(xo, Nm, fs)
    pri_lim = [item for sublist in pri_lim for item in sublist]


    engine = MCMCEngine(model, prior, transition_model, std_tr, pri_lim)

    num_chains =5
    initial_S_values = [-6, -7, -8, -5,-7.5]#,-6.5,-8.5]

    initial_conditions = []
    # xo = MDOF_LSQ.create_initial_list(Nm, S_val, f, Se=-20)
    # initial_conditions.append(xo)
    for S_val in initial_S_values:
        xo = MDOF_LSQ.create_initial_list(Nm, S_val, f, Se=-20)
        initial_conditions.append(xo)
    # start_time = time.time()

    # result = engine.run_chain(0, initial_conditions[0], iterations, s1)
    # elapsed_time = time.time() - start_time
    # print(f"Elapsed time: {elapsed_time:.2f} seconds")
    start_time = time.time()
    # results = engine.run_chains_parallel(initial_conditions, iterations, s1, num_chains)
    results = engine.run_chains_parallel2(initial_conditions, iterations, s1, num_chains)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # print_results(results)