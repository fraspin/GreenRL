import os
import re
import pickle
import copy

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.stats import t

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from auxiliarFunctions import append_

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
#%%# output parameters #########

TF_VALUES     = [243, 328, 410]
Mb_VALUES     = [100, 120, 140, 160, 180, 200]
OBJECTIVE     = 'rev'
# OBJECTIVE     = 'fair'
PARENTDIR     = 'test_experiments'
PRINT_FIGURES = 1

algorithms = ('GreenRL', 'GreenH', 'CC23', 'Random', 'Emptier', 'Optimal', 'CarbonEdge')

# color_vect_noSolver = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372)]
# color_vect_Solver   = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (0, 0, 0)]
# color_vect          = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.5, 0.5, 0.5),         (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (0, 0, 0)]

color_vect_noSolver = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (0.89, 0.46, 0.76)]
color_vect_Solver   = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (0, 0, 0), (0.89, 0.46, 0.76)]
color_vect          = color_vect_Solver # This ensures it inherits all 7 colors perfectly!

markerList  = ["o", "v", "x", "s", "D", ".", ", ", "^", "<", ">", "8", "p", "P",
               "*", "h", "H", "+", "X", "d", "/", "_", "1", "2", "3", "4"]
SMALL_SIZE  = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 12

##############################################################################
##############################################################################
##############################################################################

######## Directory and folders management
name_exp       = "tflops_rate_fairness" if OBJECTIVE == 'fair' else "tflops_rate_profit"
output_folder  = os.path.join(PARENTDIR, 'tflops_mbits')
outputFig_path = os.path.join(output_folder, name_exp)
if not os.path.exists(outputFig_path):
    os.makedirs(outputFig_path, exist_ok=True)
    
folders = [folder for folder in os.listdir(PARENTDIR) 
           if os.path.isdir(os.path.join(PARENTDIR, folder))]

experiments_dirs = {}
for tflops in TF_VALUES:
    for rate in Mb_VALUES:
        case = f"policy_{OBJECTIVE}_7_nodes_TF_{tflops}_WIR_{rate}"
        pattern = fr".*{case}$"

        # Find the matching folder
        matching_folder = next((folder for folder in folders if re.match(pattern, folder)), None)
        
        # Print the result
        if not matching_folder:
            print(f"({tflops},{rate}) -- No matching folder found.!!!!!\n")
            
        experiments_dirs[(tflops,rate)] = matching_folder


if OBJECTIVE == 'fair':
    nameAttribute = 'objFunction_Fairness'
    name_y_axis   = r"Fairness Obj. Function $\log(\bar{P}_{\rho_p}\bar{R}_{\rho_r})$"
elif OBJECTIVE == 'rev':
    nameAttribute = 'Objective_function_rev_minus_cost'
    name_y_axis   = r' Normalized profit margin ($\bar{B}$)'
        

def get_mean_and_conf(results_eval, results_eval_heuristic, results_eval_random,
                       results_eval_emptier, results_eval_state_art, results_eval_solver, 
                       results_eval_carbonedge, n_realizations, nameAttribute):
       
    data_policy    = []
    data_random    = []
    data_emptier   = []
    data_heurist   = []
    data_sota      = []
    data_solver    = [] 
    data_carbonedge = []

    for n_time in range(n_realizations):
        results_policy = results_eval[n_time]
        results_random = results_eval_random[n_time]
        results_emptier = results_eval_emptier[n_time]
        results_heurist = results_eval_heuristic[n_time]
        results_sota = results_eval_state_art[n_time]
        results_solver = results_eval_solver[n_time]
        results_carbonedge = results_eval_carbonedge[n_time]

        value_policy = results_policy[nameAttribute]
        value_random = results_random[nameAttribute]
        value_emptier = results_emptier[nameAttribute]
        value_heurist = results_heurist[nameAttribute]
        value_solver = results_solver[nameAttribute]

        # Safely handle missing benchmarks from older .pkl files
        value_sota = results_sota[nameAttribute] if results_sota else np.nan
        value_carbonedge = results_carbonedge[nameAttribute] if results_carbonedge else np.nan

        data_policy = append_(data_policy, value_policy)
        data_random = append_(data_random, value_random)
        data_emptier = append_(data_emptier, value_emptier)
        data_heurist = append_(data_heurist, value_heurist)
        data_sota = append_(data_sota, value_sota)
        data_solver = append_(data_solver, value_solver)
        data_carbonedge = append_(data_carbonedge, value_carbonedge)


    mean_policy     = np.average(data_policy)
    mean_random     = np.average(data_random)
    mean_emptier    = np.average(data_emptier)
    mean_heurist    = np.average(data_heurist)
    mean_sota       = np.average(data_sota)
    mean_solver     = np.average(data_solver)
    mean_carbonedge = np.average(data_carbonedge)

    stdv_policy     = np.std(data_policy)
    stdv_random     = np.std(data_random)
    stdv_emptier    = np.std(data_emptier)
    stdv_heurist    = np.std(data_heurist)
    stdv_sota       = np.std(data_sota)
    stdv_solver     = np.std(data_solver)
    stdv_carbonedge = np.std(data_carbonedge)

    std   = (stdv_policy, stdv_heurist, stdv_sota, stdv_random, stdv_emptier, stdv_solver, stdv_carbonedge)
    mean  = [mean_policy, mean_heurist, mean_sota, mean_random, mean_emptier, mean_solver, mean_carbonedge]
    order = ['GreenRL', 'GreenH', 'CC23', 'Random', 'Emptier', 'Optimal', 'CarbonEdge']
    
    return mean, std, order


########## Execute main loop

df_data = pd.DataFrame(columns=['tflops', 'rate', 'means'])

for thisCase, dirPath in experiments_dirs.items():
    if dirPath is None:
        print(f'({thisCase[0]},{thisCase[1]}) -- No matching folder found.!!!!! -->> Continue to next case')
        continue

    experiment_dir = os.path.join(PARENTDIR, dirPath)

    results_eval_file   = os.path.join(experiment_dir, 'results_eval.pkl')
    results_solver_file = os.path.join(experiment_dir, 'results_solver.pkl')

    with open(results_eval_file, 'rb') as mfile:
        print('mfile', mfile)
        results_eval = pickle.load(mfile)
        env1                   = pickle.load(mfile)
        results_eval_random = pickle.load(mfile)
        results_eval_emptier = pickle.load(mfile)
        results_eval_heuristic = pickle.load(mfile)

        n_rlzs = len(results_eval)

        try:
            results_eval_sota = pickle.load(mfile)
        except EOFError:
            results_eval_sota = [None] * n_rlzs

        try:
            results_eval_carbonedge = pickle.load(mfile)
        except EOFError:
            results_eval_carbonedge = [None] * n_rlzs

    with open(results_solver_file, 'rb') as mfile:
        results_solver = pickle.load(mfile)

    N_SERVERS   = len(results_eval[0]['pw_nt_sol'])
    n_rlzs_eval = len(results_eval)

    mean_res, std_res, alg_order = get_mean_and_conf(results_eval, results_eval_heuristic, 
                                           results_eval_random, results_eval_emptier, 
                                           results_eval_sota, results_solver, results_eval_carbonedge,
                                           n_rlzs_eval, nameAttribute)
    
    df_data.loc[len(df_data)] = {'tflops': thisCase[0], 'rate':thisCase[1], 'means': mean_res}
    

        
#%% ONLY FOR ALGORITHMS
algorithms = ('CC23', 'GreenH', 'GreenRL', 'CarbonEdge', 'Optimal')

## Data
rates           = sorted(df_data.rate.unique()) ## rates
subcases_tflops = sorted(df_data.tflops.unique()) ## tflops
alg_list        = [algo for algo in algorithms]

means_array = np.zeros((len(rates), len(algorithms), len(subcases_tflops)))
for idx_sub, subcase_tflops in enumerate(subcases_tflops):
    for idx_grp, group_rate in enumerate(rates):
        mean_values                      = df_data[(df_data['tflops'] == subcase_tflops) & (df_data['rate'] == group_rate)]['means'].values[0]
        alg_values_dict                  = dict(zip(alg_order, mean_values))
        ordered_values                   = [alg_values_dict[alg] for alg in alg_list]
        means_array[idx_grp, :, idx_sub] = ordered_values

#%% To plot the 5 algorithms
# %% To plot the 5 algorithms
def plot_5algos(obj_sota, obj_heurist, obj_policy, obj_carbonedge, obj_solver, style, bias):
    x = np.arange(len(obj_policy)) + bias  # range(0, len(obj_policy))
    plt.rcParams['text.usetex'] = True
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)

    line6, = ax.plot(x, obj_sota, markeredgecolor=color_vect[2], label='CC23', **style)
    line4, = ax.plot(x, obj_heurist, markeredgecolor=color_vect[1], label='GreenH', **style)
    line1, = ax.plot(x, obj_policy, markeredgecolor=color_vect[0], label='GreenRL', **style)
    line5, = ax.plot(x, obj_carbonedge, markeredgecolor=color_vect[6], label='CE25', **style)  # <-- CE25 Fix
    line7, = ax.plot(x, obj_solver, markeredgecolor=color_vect[5], label='Optimal', **style)  # <-- Optimal Fix


fig, ax = plt.subplots(figsize=(14, 9))

fontSizeBig = 22
fontSizeMedium = 19

xticks_pos = rates
xlabel = r'Mean Rate (Mbits)'
ylabel = name_y_axis

myLinestyle = ['dotted', 'solid', 'dashed']
myMarkerTyp = [markerList[0], markerList[3], markerList[4]]
tflop_Bias = np.array([-1, 0, 1]) * 0.15
default_style = {
    'markersize': 11,
    'markerfacecolor': 'none',
    'markeredgewidth': 3.8,
    'linewidth': 2,
    'linestyle': 'None',  ##myLinestyle[idx]
    'alpha': 0.8
}
plt.axhline(y=0.0, color='k', alpha=0.9, linewidth=.5, linestyle='-')

for idx, tflopsCases in enumerate(subcases_tflops):
    my_style = copy.deepcopy(default_style)
    my_style['marker'] = myMarkerTyp[idx]
    plot_5algos(means_array[:, 0, idx], means_array[:, 1, idx], means_array[:, 2, idx], means_array[:, 3, idx],
                means_array[:, 4, idx], my_style, tflop_Bias[idx])

ax.set_xlabel(xlabel, fontsize=fontSizeBig)
ax.set_ylabel(ylabel, fontsize=fontSizeBig)
ax.set_xticks(np.arange(len(rates)), xticks_pos, fontsize=fontSizeBig)
plt.tick_params(axis='y', labelsize=fontSizeBig)

legendMarkerStyle = copy.deepcopy(default_style)
legendMarkerStyle['markersize'] = 9
legendMarkerStyle['markeredgewidth'] = 3
legend_handles = [
    mpatches.Patch(color=color_vect[2], label='CC23'),
    mpatches.Patch(color=color_vect[1], label='GreenH'),
    mpatches.Patch(color=color_vect[0], label='GreenRL'),
    mpatches.Patch(color=color_vect[6], label='CE25'),  # <-- CE25 Legend Fix
    mpatches.Patch(color=color_vect[5], label='Optimal'),
    mlines.Line2D([], [], color='black', marker=myMarkerTyp[0], **legendMarkerStyle,
                  label=f'{subcases_tflops[0]} Tflops'),
    mlines.Line2D([], [], color='black', marker=myMarkerTyp[1], **legendMarkerStyle,
                  label=f'{subcases_tflops[1]} Tflops'),
    mlines.Line2D([], [], color='black', marker=myMarkerTyp[2], **legendMarkerStyle,
                  label=f'{subcases_tflops[2]} Tflops')
]
plt.legend(handles=legend_handles, ncols=2, fontsize=fontSizeMedium)
# ax.legend(loc=(0.19, 0.01))

fig.savefig(outputFig_path, bbox_inches='tight')

if PRINT_FIGURES:
    plt.show()
else:
    plt.close()
    
    