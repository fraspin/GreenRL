import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
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
### output parameters #########
exp_folder    = 'test_experiments'
output_folder = os.path.join('test_experiments', 'bar_plots_all')

name_exp      = '7_nodes'

### input parameters ##########
MAX_CYCLE_NORM_SERVER = 10
tot_time_evaluation   = 540
exp_folder            = 'test_experiments'
experiment_dir1       = 'saved_model_policy_fair_7_nodes' # PUT HERE THE EXPERIMENT YOU WANT TO PLOT (FAIRNESS P2)
experiment_dir2       = 'saved_model_policy_rev_7_nodes'  # PUT HERE THE EXPERIMENT YOU WANT TO PLOT (PROFIT  P1)
format_fig            = 'pdf'

PRINT_FIGURES       = 0
TRANSIENT_PERC      = 0.0


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

color_vect_noSolver = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372)]
color_vect_Solver   = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (0,0,0)]
color_vect_noSolver = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (.9, .9, .9)]
color_vect_Solver   = [(0.4705, 0, 0), (0.7568, 0.0705, 0.1215), (0.4471, 0.651, 0.3098), (0, 0.188, 0.286), (0.4, 0.60784, 0.7372), (0,0,0), (.9, .9, .9)]

markerList  = ["o", "v", "x", "s", "D", ".", ", ", "^", "<", ">", "8", "p", "P",
               "*", "h", "H", "+", "X", "d", "/", "_", "1", "2", "3", "4"]
confidence  = 0.95
SMALL_SIZE  = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 15

def getDataAcceptedJobs(results_eval, results_eval_heuristic, 
                        results_eval_sota, results_eval_ce, 
                        results_eval_random, results_eval_emptier, 
                        results_eval_solver, n_realizations):

    accepted_jobs_t           = []
    rejected_jobs_t           = []
    accepted_jobs_t_sota      = []
    rejected_jobs_t_sota      = []
    accepted_jobs_t_ce        = []
    rejected_jobs_t_ce        = []
    accepted_jobs_t_random    = []
    rejected_jobs_t_random    = []
    accepted_jobs_t_emptier   = []
    rejected_jobs_t_emptier   = []
    accepted_jobs_t_heuristic = []
    rejected_jobs_t_heuristic = []
    
    ## particular for policy
    rejectedPW_jobs_t          = []
    rejectedDelay_jobs_t       = []    
    interruptedPW_jobs_t       = []
    interruptedRejected_jobs_t = []
    interruptedDelayMig_jobs_t = []
    
    ## particular for sota    
    rejectedPW_jobs_t_sota    = []
    rejectedDelay_jobs_t_sota = []

    ## particular for carbonedge
    rejectedPW_jobs_t_ce    = []
    rejectedDelay_jobs_t_ce = []

    ## particular for heuristic   
    interruptedPW_jobs_t_heuristic       = []
    interruptedDelayMig_jobs_t_heuristic = [] 
    interruptedRejected_jobs_t_heuristic = []
    
    if accepted_jobs_t == []:
        if n_realizations == 1:
            appendCase = 'singleAssignment'
        else: 
            appendCase = 'startList'
    else:
            appendCase = 'append'

    def append_robust(varList, newItem, case):
        if case == 'singleAssignment': ## no vector --  direct assignment
            varList = newItem
        elif case == 'startList': ## initiate the list
            varList = [newItem]
        elif case == 'append':
            varList.append(newItem)
        return varList

    for n_time in range(n_realizations):
        results_jobs_t_rl        = results_eval[n_time]
        results_jobs_t_random    = results_eval_random[n_time]
        results_jobs_t_emptier   = results_eval_emptier[n_time]
        results_jobs_t_heuristic = results_eval_heuristic[n_time]
        results_jobs_t_sota      = results_eval_sota[n_time]
        results_jobs_t_ce        = results_eval_ce[n_time]

        ## tot = accepted_jobs_t + rejection_own + rejected_jobs_power_t + rejected_jobs_delay_t
        accepted_jobs_t            = append_robust(accepted_jobs_t, results_jobs_t_rl['accepted_jobs_t'], appendCase)
        rejected_jobs_t            = append_robust(rejected_jobs_t, results_jobs_t_rl['rejection_own'], appendCase) ## rejected on purpose, no error
        rejectedPW_jobs_t          = append_robust(rejectedPW_jobs_t, results_jobs_t_rl['rejected_jobs_power_t'], appendCase)
        rejectedDelay_jobs_t       = append_robust(rejectedDelay_jobs_t, results_jobs_t_rl['rejected_jobs_delay_t'], appendCase)
        
        interruptedPW_jobs_t       = append_robust(interruptedPW_jobs_t, results_jobs_t_rl['interruptedpower_t'], appendCase)
        interruptedRejected_jobs_t = append_robust(interruptedRejected_jobs_t, results_jobs_t_rl['interrupted_rejected_t'], appendCase)
        interruptedDelayMig_jobs_t = append_robust(interruptedDelayMig_jobs_t, results_jobs_t_rl['interruptedelay_t'], appendCase)
        
        accepted_jobs_t_random     = append_robust(accepted_jobs_t_random, results_jobs_t_random['accepted_jobs_t'], appendCase)
        rejected_jobs_t_random     = append_robust(rejected_jobs_t_random, results_jobs_t_random['rejected_jobs_t'], appendCase)
        
        accepted_jobs_t_emptier    = append_robust(accepted_jobs_t_emptier, results_jobs_t_emptier['accepted_jobs_t'], appendCase)
        rejected_jobs_t_emptier    = append_robust(rejected_jobs_t_emptier, results_jobs_t_emptier['rejected_jobs_t'], appendCase)
        
        accepted_jobs_t_heuristic  = append_robust(accepted_jobs_t_heuristic, results_jobs_t_heuristic['accepted_jobs_t'], appendCase)
        rejected_jobs_t_heuristic  = append_robust(rejected_jobs_t_heuristic, results_jobs_t_heuristic['rejected_jobs_t'], appendCase)
        
        interruptedPW_jobs_t_heuristic       = append_robust(interruptedPW_jobs_t_heuristic, results_jobs_t_heuristic['interruptedpower_t'], appendCase)
        interruptedRejected_jobs_t_heuristic = append_robust(interruptedRejected_jobs_t_heuristic, results_jobs_t_heuristic['interrupted_rejected_t'], appendCase)        
        interruptedDelayMig_jobs_t_heuristic = append_robust(interruptedDelayMig_jobs_t_heuristic, results_jobs_t_heuristic['interruptedelay_t'], appendCase)
        
        accepted_jobs_t_sota         = append_robust(accepted_jobs_t_sota, results_jobs_t_sota['accepted_jobs_t'], appendCase)
        rejected_jobs_t_sota         = append_robust(rejected_jobs_t_sota, results_jobs_t_sota['rejection_own'], appendCase) ## rejected on purpose, no error
        
        rejectedPW_jobs_t_sota       = append_robust(rejectedPW_jobs_t_sota, results_jobs_t_sota['rejected_jobs_power_t'], appendCase)
        rejectedDelay_jobs_t_sota    = append_robust(rejectedDelay_jobs_t_sota, results_jobs_t_sota['deadline_violation_rejected_new_jobs'], appendCase)

        accepted_jobs_t_ce           = append_robust(accepted_jobs_t_ce, results_jobs_t_ce['accepted_jobs_t'], appendCase)
        rejected_jobs_t_ce           = append_robust(rejected_jobs_t_ce, results_jobs_t_ce['rejection_own'], appendCase) ## rejected on purpose, no error
        rejectedPW_jobs_t_ce         = append_robust(rejectedPW_jobs_t_ce, results_jobs_t_ce['rejected_jobs_power_t'], appendCase)
        rejectedDelay_jobs_t_ce      = append_robust(rejectedDelay_jobs_t_ce, results_jobs_t_ce['deadline_violation_rejected_new_jobs'], appendCase)
        

    dict_vals = dict()
    dict_vals['value_accepted_jobs_policy']    = np.sum([np.sum(sublist) for sublist in accepted_jobs_t])
    dict_vals['value_rejected_jobs_policy']    = np.sum([np.sum(sublist) for sublist in rejected_jobs_t])
    dict_vals['value_accepted_jobs_random']    = np.sum([np.sum(sublist) for sublist in accepted_jobs_t_random])
    dict_vals['value_rejected_jobs_random']    = np.sum([np.sum(sublist) for sublist in rejected_jobs_t_random])
    dict_vals['value_accepted_jobs_emptier']   = np.sum([np.sum(sublist) for sublist in accepted_jobs_t_emptier])
    dict_vals['value_rejected_jobs_emptier']   = np.sum([np.sum(sublist) for sublist in rejected_jobs_t_emptier])
    dict_vals['value_accepted_jobs_heuristic'] = np.sum([np.sum(sublist) for sublist in accepted_jobs_t_heuristic])
    dict_vals['value_rejected_jobs_heuristic'] = np.sum([np.sum(sublist) for sublist in rejected_jobs_t_heuristic])
    dict_vals['value_accepted_jobs_sota']      = np.sum([np.sum(sublist) for sublist in accepted_jobs_t_sota])
    dict_vals['value_rejected_jobs_sota']      = np.sum([np.sum(sublist) for sublist in rejected_jobs_t_sota])
    dict_vals['value_accepted_jobs_ce']        = np.sum([np.sum(sublist) for sublist in accepted_jobs_t_ce])
    dict_vals['value_rejected_jobs_ce']        = np.sum([np.sum(sublist) for sublist in rejected_jobs_t_ce])

    dict_vals['value_rejectedPW_jobs_policy']          = np.sum([np.sum(sublist) for sublist in rejectedPW_jobs_t])
    dict_vals['value_rejectedDelay_jobs_policy']       = np.sum([np.sum(sublist) for sublist in rejectedDelay_jobs_t])
    dict_vals['value_interruptedPW_jobs_policy']       = np.sum([np.sum(sublist) for sublist in interruptedPW_jobs_t])
    dict_vals['value_interruptedRejected_jobs_policy'] = np.sum([np.sum(sublist) for sublist in interruptedRejected_jobs_t])
    dict_vals['value_interruptedDelayMig_jobs_policy'] = np.sum([np.sum(sublist) for sublist in interruptedDelayMig_jobs_t])

    dict_vals['value_interruptedPW_jobs_heuristic']       = np.sum([np.sum(sublist) for sublist in interruptedPW_jobs_t_heuristic])
    dict_vals['value_interruptedRejected_jobs_heuristic'] = np.sum([np.sum(sublist) for sublist in interruptedRejected_jobs_t_heuristic])
    dict_vals['value_interruptedDelayMig_jobs_heuristic'] = np.sum([np.sum(sublist) for sublist in interruptedDelayMig_jobs_t_heuristic])
    
    dict_vals['value_rejectedPW_jobs_sota']    = np.sum([np.sum(sublist) for sublist in rejectedPW_jobs_t_sota])
    dict_vals['value_rejectedDelay_jobs_sota'] = np.sum([np.sum(sublist) for sublist in rejectedDelay_jobs_t_sota])

    dict_vals['value_rejectedPW_jobs_ce']    = np.sum([np.sum(sublist) for sublist in rejectedPW_jobs_t_ce])
    dict_vals['value_rejectedDelay_jobs_ce'] = np.sum([np.sum(sublist) for sublist in rejectedDelay_jobs_t_ce])

    accepted_jobs_t_solver  = [] if results_eval_solver != None else None
    rejected_jobs_t_solver  = [] if results_eval_solver != None else None

    if results_eval_solver != None:
       for n_time in range(n_realizations):
            results_solver  = results_eval_solver[n_time]

            accepted_jobs_t_solver = append_robust(accepted_jobs_t_solver, results_solver['accepted_jobs_t'], appendCase)
            rejected_jobs_t_solver = append_robust(rejected_jobs_t_solver, results_solver['rejected_jobs_t'], appendCase)
            

       dict_vals['value_accepted_jobs_solver'] = np.sum([np.sum(sublist) for sublist in accepted_jobs_t_solver   ])
       dict_vals['value_rejected_jobs_solver'] = np.sum([np.sum(sublist) for sublist in rejected_jobs_t_solver   ])

    return dict_vals


def plot_jobs_accepted_rejected_all(dict_values1, dict_values2, 
                                    experiment_dir, ylabel, namefig):

    nameFig = namefig + 'noSolver.pdf' if 'value_accepted_jobs_solver' not in dict_values1.keys() else namefig
    path_figure = os.path.join('.', experiment_dir,  nameFig)

    value_accepted_jobs_policy1    = dict_values1['value_accepted_jobs_policy']
    value_accepted_jobs_policy2    = dict_values2['value_accepted_jobs_policy']
    value_rejected_jobs_policy1    = dict_values1['value_rejected_jobs_policy']
    value_rejected_jobs_policy2    = dict_values2['value_rejected_jobs_policy']
    value_accepted_jobs_random1    = dict_values1['value_accepted_jobs_random']
    value_accepted_jobs_random2    = dict_values2['value_accepted_jobs_random']
    value_rejected_jobs_random1    = dict_values1['value_rejected_jobs_random']
    value_rejected_jobs_random2    = dict_values2['value_rejected_jobs_random']
    value_accepted_jobs_emptier1   = dict_values1['value_accepted_jobs_emptier']
    value_accepted_jobs_emptier2   = dict_values2['value_accepted_jobs_emptier']
    value_rejected_jobs_emptier1   = dict_values1['value_rejected_jobs_emptier']
    value_rejected_jobs_emptier2   = dict_values2['value_rejected_jobs_emptier']
    value_accepted_jobs_heuristic1 = dict_values1['value_accepted_jobs_heuristic']
    value_accepted_jobs_heuristic2 = dict_values2['value_accepted_jobs_heuristic']
    value_rejected_jobs_heuristic1 = dict_values1['value_rejected_jobs_heuristic']
    value_rejected_jobs_heuristic2 = dict_values2['value_rejected_jobs_heuristic']
    value_accepted_jobs_sota1      = dict_values1['value_accepted_jobs_sota']
    value_accepted_jobs_sota2      = dict_values2['value_accepted_jobs_sota']
    value_rejected_jobs_sota1      = dict_values1['value_rejected_jobs_sota']
    value_rejected_jobs_sota2      = dict_values2['value_rejected_jobs_sota']
    value_accepted_jobs_ce1        = dict_values1['value_accepted_jobs_ce']
    value_accepted_jobs_ce2        = dict_values2['value_accepted_jobs_ce']
    value_rejected_jobs_ce1        = dict_values1['value_rejected_jobs_ce']
    value_rejected_jobs_ce2        = dict_values2['value_rejected_jobs_ce']

    value_rejectedPW_jobs_policy1          = dict_values1['value_rejectedPW_jobs_policy']
    value_rejectedPW_jobs_policy2          = dict_values2['value_rejectedPW_jobs_policy']
    value_rejectedDelay_jobs_policy1       = dict_values1['value_rejectedDelay_jobs_policy']
    value_rejectedDelay_jobs_policy2       = dict_values2['value_rejectedDelay_jobs_policy']
    value_interruptedRejected_jobs_policy1 = dict_values1['value_interruptedRejected_jobs_policy']
    value_interruptedRejected_jobs_policy2 = dict_values2['value_interruptedRejected_jobs_policy']
    value_interruptedPW_jobs_policy1       = dict_values1['value_interruptedPW_jobs_policy']
    value_interruptedPW_jobs_policy2       = dict_values2['value_interruptedPW_jobs_policy']
    value_interruptedDelayMig_jobs_policy1 = dict_values1['value_interruptedDelayMig_jobs_policy']
    value_interruptedDelayMig_jobs_policy2 = dict_values2['value_interruptedDelayMig_jobs_policy']                   
                    
    value_interruptedPW_jobs_heuristic1       = dict_values1['value_interruptedPW_jobs_heuristic']
    value_interruptedPW_jobs_heuristic2       = dict_values2['value_interruptedPW_jobs_heuristic']    
    value_interruptedDelayMig_jobs_heuristic1 = dict_values1['value_interruptedDelayMig_jobs_heuristic']
    value_interruptedDelayMig_jobs_heuristic2 = dict_values2['value_interruptedDelayMig_jobs_heuristic']
    value_interruptedRejected_jobs_heuristic1 = dict_values1['value_interruptedRejected_jobs_heuristic']
    value_interruptedRejected_jobs_heuristic2 = dict_values2['value_interruptedRejected_jobs_heuristic']
    
    value_rejectedPW_jobs_sota1         = dict_values1['value_rejectedPW_jobs_sota']
    value_rejectedPW_jobs_sota2         = dict_values2['value_rejectedPW_jobs_sota']
    value_rejectedDelay_jobs_sota1      = dict_values1['value_rejectedDelay_jobs_sota']
    value_rejectedDelay_jobs_sota2      = dict_values2['value_rejectedDelay_jobs_sota']

    value_rejectedPW_jobs_ce1         = dict_values1['value_rejectedPW_jobs_ce']
    value_rejectedPW_jobs_ce2         = dict_values2['value_rejectedPW_jobs_ce']
    value_rejectedDelay_jobs_ce1      = dict_values1['value_rejectedDelay_jobs_ce']
    value_rejectedDelay_jobs_ce2      = dict_values2['value_rejectedDelay_jobs_ce']
    
    if 'value_accepted_jobs_solver'  in dict_values1.keys()  != None:
        value_accepted_jobs_solver1 = dict_values1['value_accepted_jobs_solver']
        value_accepted_jobs_solver2 = dict_values2['value_accepted_jobs_solver']
        value_rejected_jobs_solver1 = dict_values1['value_rejected_jobs_solver']
        value_rejected_jobs_solver2 = dict_values2['value_rejected_jobs_solver']
        
    color_vect = color_vect_noSolver

    #### Values for CASE 1 #########################################
    
    norm_v1 = value_accepted_jobs_policy1   + value_rejected_jobs_policy1  \
              + value_rejectedPW_jobs_policy1 + value_rejectedDelay_jobs_policy1

    norm_v1_solver = value_accepted_jobs_solver1 + value_rejected_jobs_solver1
    
    j_rejected1 = (value_rejected_jobs_policy1, 0, value_rejected_jobs_sota1, value_rejected_jobs_ce1, 0, 0)/norm_v1

    j_rejectedError1  = (value_rejectedPW_jobs_policy1 + value_rejectedDelay_jobs_policy1, 
                         value_rejected_jobs_heuristic1, 
                         value_rejectedPW_jobs_sota1 + value_rejectedDelay_jobs_sota1, 
                         value_rejectedPW_jobs_ce1 + value_rejectedDelay_jobs_ce1, 
                         value_rejected_jobs_random1, value_rejected_jobs_emptier1)/norm_v1
    
    j_interruptedAll1   = (value_interruptedPW_jobs_policy1 + value_interruptedRejected_jobs_policy1 + value_interruptedDelayMig_jobs_policy1,
                          value_interruptedPW_jobs_heuristic1 + value_interruptedDelayMig_jobs_heuristic1 + value_interruptedRejected_jobs_heuristic1, 
                          0, 0, 0, 0)/norm_v1
    
    j_accepted1 = (value_accepted_jobs_policy1, value_accepted_jobs_heuristic1, 
                   value_accepted_jobs_sota1,   value_accepted_jobs_ce1, 
                   value_accepted_jobs_random1, value_accepted_jobs_emptier1)/norm_v1

    j_accepted1 = tuple(np.subtract(j_accepted1, j_interruptedAll1))  

    j_accepted1       = tuple(j_accepted1)       + (value_accepted_jobs_solver1/norm_v1_solver,)
    j_rejected1       = tuple(j_rejected1)       + (value_rejected_jobs_solver1/norm_v1_solver,)
    j_rejectedError1  = tuple(j_rejectedError1)  + (0,)
    j_interruptedAll1 = tuple(j_interruptedAll1) + (0,)


    #### Values for CASE 2 #########################################
    
    norm_v2 = value_accepted_jobs_policy2   + value_rejected_jobs_policy2  \
              + value_rejectedPW_jobs_policy2 + value_rejectedDelay_jobs_policy2

    norm_v2_solver = value_accepted_jobs_solver2 + value_rejected_jobs_solver2
    
    j_rejected2 = (value_rejected_jobs_policy2, 0, value_rejected_jobs_sota2, value_rejected_jobs_ce2, 0, 0)/norm_v2

    j_rejectedError2  = (value_rejectedPW_jobs_policy2 + value_rejectedDelay_jobs_policy2, 
                         value_rejected_jobs_heuristic2, 
                         value_rejectedPW_jobs_sota2 + value_rejectedDelay_jobs_sota2, 
                         value_rejectedPW_jobs_ce2 + value_rejectedDelay_jobs_ce2, 
                         value_rejected_jobs_random2, value_rejected_jobs_emptier2)/norm_v2
    
    j_interruptedAll2   = (value_interruptedPW_jobs_policy2 + value_interruptedRejected_jobs_policy2 + value_interruptedDelayMig_jobs_policy2,
                          value_interruptedPW_jobs_heuristic2 + value_interruptedDelayMig_jobs_heuristic2 + value_interruptedRejected_jobs_heuristic2, 
                          0, 0, 0, 0)/norm_v2

    j_accepted2 = (value_accepted_jobs_policy2, value_accepted_jobs_heuristic2, 
                   value_accepted_jobs_sota2, value_accepted_jobs_ce2, 
                   value_accepted_jobs_random2, value_accepted_jobs_emptier2)/norm_v2

    j_accepted2 = tuple(np.subtract(j_accepted2, j_interruptedAll2))  
    j_accepted2       = tuple(j_accepted2)       + (value_accepted_jobs_solver2/norm_v2_solver,)
    j_rejected2       = tuple(j_rejected2)       + (value_rejected_jobs_solver2/norm_v2_solver,)
    j_rejectedError2  = tuple(j_rejectedError2)     + (0,)
    j_interruptedAll2 = tuple(j_interruptedAll2) + (0,)

    #### ADAPTATION FOR SINGLE CASE #########################################
    print(j_accepted1[0], j_accepted2[0], j_accepted1[1:])

    j_accepted1       = tuple([j_accepted1[0]]       + [j_accepted2[0]]       + list(j_accepted1[1:]))
    j_rejected1       = tuple([j_rejected1[0]]       + [j_rejected2[0]]       + list(j_rejected1[1:]))
    j_rejectedError1  = tuple([j_rejectedError1[0]]  + [j_rejectedError2[0]]  + list(j_rejectedError1[1:]))
    j_interruptedAll1 = tuple([j_interruptedAll1[0]] + [j_interruptedAll2[0]] + list(j_interruptedAll1[1:]))
    
    j_accepted1       = tuple(j_accepted1)       + (j_accepted2[-1],)
    j_rejected1       = tuple(j_rejected1)       + (j_rejected2[-1],)
    j_rejectedError1  = tuple(j_rejectedError1)  + (j_rejectedError2[-1],)
    j_interruptedAll1 = tuple(j_interruptedAll1) + (j_interruptedAll2[-1],)
    
    #### PLOTTING #########################################

    legend_alg_b = ('GreenRL\n Fairness', 'GreenRL \n Profit', 'GreenH',
             'CC23', 'CE25', 'Random', 'Emptier', 'Optimal \n Fairness', 'Optimal \n Profit')   

    bottom_interruptAll1  = tuple([j1 - j2 - j3 for j1, j2, j3 in zip(j_accepted1, j_rejectedError1, j_interruptedAll1)])
    bottom_interruptAll1  = tuple(j_accepted1)
    bottom_rejectedError1 = tuple([j1 + j2 for j1, j2 in zip(bottom_interruptAll1, j_interruptedAll1)])
    bottom_rejected_OK1   = tuple([j1 + j2 for j1, j2 in zip(bottom_rejectedError1, j_rejectedError1)])

    bottom_interruptAll2  = tuple([j1 - j2 - j3 for j1, j2, j3 in zip(j_accepted2, j_rejectedError2, j_interruptedAll2)])
    bottom_interruptAll2  = tuple(j_accepted2)
    bottom_rejectedError2 = tuple([j1 + j2 for j1, j2 in zip(bottom_interruptAll2, j_interruptedAll2)])
    bottom_rejected_OK2   = tuple([j1 + j2 for j1, j2 in zip(bottom_rejectedError2, j_rejectedError2)])

    fig, axsplt1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 4)
    fig.set_dpi(300)
    plt.rcParams['text.usetex'] = True

    j_accepted       = [j1 for j1 in j_accepted1]
    j_interruptedAll = [j1 for j1 in j_interruptedAll1]
    j_rejectedError  = [j1 for j1 in j_rejectedError1]
    j_rejected       = [j1 for j1 in j_rejected1]

    bottom_interruptAll  = [j1 for j1 in bottom_interruptAll1]
    bottom_rejectedError = [j1 for j1 in bottom_rejectedError1]
    bottom_rejectedOK    = [j1 for j1 in bottom_rejected_OK1]
         

    print(np.array(j_accepted)+np.array(j_interruptedAll)+np.array(j_rejectedError)+np.array(j_rejected))
    print(np.array(j_interruptedAll))
    print(np.array(j_rejectedError))
    print(np.array(j_rejected))
    
    y_pos = np.arange(len(legend_alg_b))
    plt.rcParams['text.usetex'] = True
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=MEDIUM_SIZE)
    plt.rc('ytick', labelsize=MEDIUM_SIZE)
    plt.rcParams['hatch.linewidth'] = 2.0  # Adjust the value as needed

    axsplt1.set_ylabel(ylabel, fontsize = MEDIUM_SIZE)
    p1 = plt.bar(y_pos, j_accepted, linewidth=0,
                 color = color_vect[3],  hatch = ['/', None, None, None, None, None, None,'/', None], edgecolor = 'white', align='center', alpha=0.7 )
    p2 = plt.bar(y_pos, j_interruptedAll, bottom=bottom_interruptAll, linewidth=0,
                 color = color_vect[0], hatch = ['/', None, None, None, None, None, None,'/', None], edgecolor = 'white')
    p4 = plt.bar(y_pos, j_rejectedError,  bottom=bottom_rejectedError, linewidth=0,
                 color = color_vect[1], hatch = ['/', None, None, None, None, None, None,'/', None], edgecolor = 'white')
    p5 = plt.bar(y_pos, j_rejected,  bottom=bottom_rejectedOK,  linewidth=0,
                 color = color_vect[2], hatch = ['/', None, None, None, None, None, None,'/', None], edgecolor = 'white')






    plt.legend((p5[0], p4[0], p2[0], p1[0]), ('Non-offloaded jobs', 'Forcedly rejected jobs', 'Interrupted jobs',  ' Accepted jobs'), fontsize=17)
    plt.ylabel(ylabel)
    plt.xticks(y_pos, legend_alg_b)

    plt.xlim((-0.5,8.5))
    fig.savefig(path_figure, bbox_inches='tight')
    # plt.show()
    

def plot_pw_alg(results_eval, results_eval_heuristic, results_eval_random,
                       results_eval_emptier, results_eval_solver1, results_eval_sota, results_eval_ce,
                       results_eval2, results_eval_heuristic2, results_eval_random2,
                       results_eval_emptier2, results_eval_solver2, results_eval_sota2, results_eval_ce2,
                       n_realizations, TRANSIENT, experiment_dir, N_SERVERS, MAX_CYCLE, yaxis_name, namefig):

    ylabel  = yaxis_name
    nameFig = namefig + 'noSolver.pdf' if results_eval_solver1 == None else namefig
    path_figure = os.path.join('.', experiment_dir,  nameFig)

    nameFig = 'energyConsumption_'

    green_energy_avail_t = []
    brown_energy_used_t  = []
    green_energy_used_t  = []

    green_energy_avail_t_random = []
    brown_energy_used_t_random  = []
    green_energy_used_t_random  = []

    green_energy_avail_t_emptier = []
    green_energy_used_t_emptier  = []
    brown_energy_used_t_emptier  = []

    green_energy_avail_t_heuristic = []
    brown_energy_used_t_heuristic  = []
    green_energy_used_t_heuristic  = []

    green_energy_avail_t_sota = []
    brown_energy_used_t_sota  = []
    green_energy_used_t_sota  = []

    green_energy_avail_t_ce = []
    brown_energy_used_t_ce  = []
    green_energy_used_t_ce  = []

    green_energy_avail_t2 = []
    brown_energy_used_t2  = []
    green_energy_used_t2  = []

    green_energy_avail_t_random2 = []
    brown_energy_used_t_random2  = []
    green_energy_used_t_random2  = []

    green_energy_avail_t_emptier2 = []
    green_energy_used_t_emptier2  = []
    brown_energy_used_t_emptier2  = []

    green_energy_avail_t_heuristic2 = []
    brown_energy_used_t_heuristic2  = []
    green_energy_used_t_heuristic2  = []

    green_energy_avail_t_sota2 = []
    brown_energy_used_t_sota2  = []
    green_energy_used_t_sota2  = []

    green_energy_avail_t_ce2 = []
    brown_energy_used_t_ce2  = []
    green_energy_used_t_ce2  = []

    max_pw_server = MAX_CYCLE[0]

    def getUsedGreenPW(df):
        return  (np.array(df['pw_nt_sol'])[:,TRANSIENT:]-np.array(df['pw_nt_br_sol'])[:,TRANSIENT:]) #df['pw_nt_gr_sol'] -

    duration_realization     = len(results_eval[0]['pw_nt_gr_sol'][0])
    duration_realization_solver1 = len(results_eval_solver1[0]['pw_nt_gr_sol'][0])
    duration_realization_solver2 = len(results_eval_solver2[0]['pw_nt_gr_sol'][0])
    print('duration_realization', duration_realization)
    print('duration_realization_solver1', duration_realization_solver1)
    print('duration_realization_solver2', duration_realization_solver2)

    for n_eval in range(n_realizations):
        results_jobs_t_rl        = results_eval[n_eval]
        results_jobs_t_random    = results_eval_random[n_eval]
        results_jobs_t_emptier   = results_eval_emptier[n_eval]
        results_jobs_t_heuristic = results_eval_heuristic[n_eval]
        results_jobs_t_sota      = results_eval_sota[n_eval]
        results_jobs_t_ce        = results_eval_ce[n_eval]

        results_jobs_t_rl2        = results_eval2[n_eval]
        results_jobs_t_random2    = results_eval_random2[n_eval]
        results_jobs_t_emptier2   = results_eval_emptier2[n_eval]
        results_jobs_t_heuristic2 = results_eval_heuristic2[n_eval]
        results_jobs_t_sota2      = results_eval_sota2[n_eval]
        results_jobs_t_ce2        = results_eval_ce2[n_eval]

        if green_energy_used_t_heuristic == [] and n_realizations == 1:

            green_energy_avail_t_heuristic =  np.sum(np.array(results_jobs_t_heuristic['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_heuristic  = np.sum(getUsedGreenPW(results_jobs_t_heuristic))
            brown_energy_used_t_heuristic  =  np.sum(np.array(results_jobs_t_heuristic['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_random    =  np.sum(np.array(results_jobs_t_random['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_random     =  np.sum(getUsedGreenPW(results_jobs_t_random))
            brown_energy_used_t_random     =  np.sum(np.array(results_jobs_t_random['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_emptier   =  np.sum(np.array(results_jobs_t_emptier['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_emptier    =  np.sum(getUsedGreenPW(results_jobs_t_emptier))
            brown_energy_used_t_emptier    =  np.sum(np.array(results_jobs_t_emptier['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t           =  np.sum(np.array(results_jobs_t_rl['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t            =  np.sum(getUsedGreenPW(results_jobs_t_rl))
            brown_energy_used_t            =  np.sum(np.array(results_jobs_t_rl['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_sota      =  np.sum(np.array(results_jobs_t_sota['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_sota       =  np.sum(getUsedGreenPW(results_jobs_t_sota))
            brown_energy_used_t_sota       =  np.sum(np.array(results_jobs_t_sota['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_ce        =  np.sum(np.array(results_jobs_t_ce['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_ce         =  np.sum(getUsedGreenPW(results_jobs_t_ce))
            brown_energy_used_t_ce         =  np.sum(np.array(results_jobs_t_ce['pw_nt_br_sol'])[:,TRANSIENT:])

            green_energy_avail_t_heuristic2 =  np.sum(np.array(results_jobs_t_heuristic2['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_heuristic2  =  np.sum(getUsedGreenPW(results_jobs_t_heuristic2))
            brown_energy_used_t_heuristic2  =  np.sum(np.array(results_jobs_t_heuristic2['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_random2    =  np.sum(np.array(results_jobs_t_random2['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_random2     =  np.sum(getUsedGreenPW(results_jobs_t_random2))
            brown_energy_used_t_random2     =  np.sum(np.array(results_jobs_t_random2['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_emptier2   =  np.sum(np.array(results_jobs_t_emptier2['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_emptier2    =  np.sum(getUsedGreenPW(results_jobs_t_emptier2))
            brown_energy_used_t_emptier2    =  np.sum(np.array(results_jobs_t_emptier2['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t2           =  np.sum(np.array(results_jobs_t_rl2['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t2            =  np.sum(getUsedGreenPW(results_jobs_t_rl2))
            brown_energy_used_t2            =  np.sum(np.array(results_jobs_t_rl2['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_sota2      =  np.sum(np.array(results_jobs_t_sota2['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_sota2       =  np.sum(getUsedGreenPW(results_jobs_t_sota2))
            brown_energy_used_t_sota2       =  np.sum(np.array(results_jobs_t_sota2['pw_nt_br_sol'])[:,TRANSIENT:])
            green_energy_avail_t_ce2        =  np.sum(np.array(results_jobs_t_ce2['pw_nt_gr_sol'])[:,TRANSIENT:])
            green_energy_used_t_ce2         =  np.sum(getUsedGreenPW(results_jobs_t_ce2))
            brown_energy_used_t_ce2         =  np.sum(np.array(results_jobs_t_ce2['pw_nt_br_sol'])[:,TRANSIENT:])

        elif green_energy_used_t_heuristic == [] and n_realizations > 0:

            green_energy_used_t_heuristic   = [ np.sum(getUsedGreenPW(results_jobs_t_heuristic))]
            green_energy_avail_t_heuristic  = [ np.sum(np.array(results_jobs_t_heuristic['pw_nt_gr_sol'])[:,TRANSIENT:])]
            brown_energy_used_t_heuristic   = [ np.sum(np.array(results_jobs_t_heuristic['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_random     = [ np.sum(np.array(results_jobs_t_random['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_random      = [ np.sum(getUsedGreenPW(results_jobs_t_random))]
            brown_energy_used_t_random      = [ np.sum(np.array(results_jobs_t_random['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_emptier    = [ np.sum(np.array(results_jobs_t_emptier['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_emptier     = [ np.sum(getUsedGreenPW(results_jobs_t_emptier))]
            brown_energy_used_t_emptier     = [ np.sum(np.array(results_jobs_t_emptier['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t            = [ np.sum(np.array(results_jobs_t_rl['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t             = [ np.sum(getUsedGreenPW(results_jobs_t_rl))]
            brown_energy_used_t             = [ np.sum(np.array(results_jobs_t_rl['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_sota       = [ np.sum(np.array(results_jobs_t_sota['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_sota        = [ np.sum(getUsedGreenPW(results_jobs_t_sota))]
            brown_energy_used_t_sota        = [ np.sum(np.array(results_jobs_t_sota['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_ce         = [ np.sum(np.array(results_jobs_t_ce['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_ce          = [ np.sum(getUsedGreenPW(results_jobs_t_ce))]
            brown_energy_used_t_ce          = [ np.sum(np.array(results_jobs_t_ce['pw_nt_br_sol'])[:,TRANSIENT:])]

            green_energy_used_t_heuristic2  = [ np.sum(getUsedGreenPW(results_jobs_t_heuristic2))]
            green_energy_avail_t_heuristic2 = [ np.sum(np.array(results_jobs_t_heuristic2['pw_nt_gr_sol'])[:,TRANSIENT:])]
            brown_energy_used_t_heuristic2  = [ np.sum(np.array(results_jobs_t_heuristic2['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_random2    = [ np.sum(np.array(results_jobs_t_random2['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_random2     = [ np.sum(getUsedGreenPW(results_jobs_t_random2))]
            brown_energy_used_t_random2     = [ np.sum(np.array(results_jobs_t_random2['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_emptier2   = [ np.sum(np.array(results_jobs_t_emptier2['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_emptier2    = [ np.sum(getUsedGreenPW(results_jobs_t_emptier2))]
            brown_energy_used_t_emptier2    = [ np.sum(np.array(results_jobs_t_emptier2['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t2           = [ np.sum(np.array(results_jobs_t_rl2['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t2            = [ np.sum(getUsedGreenPW(results_jobs_t_rl2))]
            brown_energy_used_t2            = [ np.sum(np.array(results_jobs_t_rl2['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_sota2      = [ np.sum(np.array(results_jobs_t_sota2['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_sota2       = [ np.sum(getUsedGreenPW(results_jobs_t_sota2))]
            brown_energy_used_t_sota2       = [ np.sum(np.array(results_jobs_t_sota2['pw_nt_br_sol'])[:,TRANSIENT:])]
            green_energy_avail_t_ce2        = [ np.sum(np.array(results_jobs_t_ce2['pw_nt_gr_sol'])[:,TRANSIENT:])]
            green_energy_used_t_ce2         = [ np.sum(getUsedGreenPW(results_jobs_t_ce2))]
            brown_energy_used_t_ce2         = [ np.sum(np.array(results_jobs_t_ce2['pw_nt_br_sol'])[:,TRANSIENT:])]

        else:
            green_energy_avail_t_heuristic.append( np.sum(np.array(results_jobs_t_heuristic['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_heuristic.append( np.sum(getUsedGreenPW(results_jobs_t_heuristic)))
            brown_energy_used_t_heuristic.append( np.sum(np.array(results_jobs_t_heuristic['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_random.append( np.sum(np.array(results_jobs_t_random['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_random.append( np.sum(getUsedGreenPW(results_jobs_t_random)))
            brown_energy_used_t_random.append( np.sum(np.array(results_jobs_t_random['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_emptier.append( np.sum(np.array(results_jobs_t_emptier['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_emptier.append( np.sum(getUsedGreenPW(results_jobs_t_emptier)))
            brown_energy_used_t_emptier.append( np.sum(np.array(results_jobs_t_emptier['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t.append( np.sum(np.array(results_jobs_t_rl['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t.append( np.sum(getUsedGreenPW(results_jobs_t_rl)))
            brown_energy_used_t.append( np.sum(np.array(results_jobs_t_rl['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_sota.append( np.sum(np.array(results_jobs_t_sota['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_sota.append( np.sum(getUsedGreenPW(results_jobs_t_sota)))
            brown_energy_used_t_sota.append( np.sum(np.array(results_jobs_t_sota['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_ce.append( np.sum(np.array(results_jobs_t_ce['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_ce.append( np.sum(getUsedGreenPW(results_jobs_t_ce)))
            brown_energy_used_t_ce.append( np.sum(np.array(results_jobs_t_ce['pw_nt_br_sol'])[:,TRANSIENT:]))

            green_energy_avail_t_heuristic2.append( np.sum(np.array(results_jobs_t_heuristic2['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_heuristic2.append( np.sum(getUsedGreenPW(results_jobs_t_heuristic2)))
            brown_energy_used_t_heuristic2.append( np.sum(np.array(results_jobs_t_heuristic2['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_random2.append( np.sum(np.array(results_jobs_t_random2['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_random2.append( np.sum(getUsedGreenPW(results_jobs_t_random2)))
            brown_energy_used_t_random2.append( np.sum(np.array(results_jobs_t_random2['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_emptier2.append( np.sum(np.array(results_jobs_t_emptier2['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_emptier2.append( np.sum(getUsedGreenPW(results_jobs_t_emptier2)))
            brown_energy_used_t_emptier2.append( np.sum(np.array(results_jobs_t_emptier2['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t2.append( np.sum(np.array(results_jobs_t_rl2['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t2.append( np.sum(getUsedGreenPW(results_jobs_t_rl2)))
            brown_energy_used_t2.append( np.sum(np.array(results_jobs_t_rl2['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_sota2.append( np.sum(np.array(results_jobs_t_sota2['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_sota2.append( np.sum(getUsedGreenPW(results_jobs_t_sota2)))
            brown_energy_used_t_sota2.append( np.sum(np.array(results_jobs_t_sota2['pw_nt_br_sol'])[:,TRANSIENT:]))
            green_energy_avail_t_ce2.append( np.sum(np.array(results_jobs_t_ce2['pw_nt_gr_sol'])[:,TRANSIENT:]))
            green_energy_used_t_ce2.append( np.sum(getUsedGreenPW(results_jobs_t_ce2)))
            brown_energy_used_t_ce2.append( np.sum(np.array(results_jobs_t_ce2['pw_nt_br_sol'])[:,TRANSIENT:]))


    value_green_avail = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t]))
    value_green_used  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t]))
    value_brown_used  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t]))

    value_green_avail_random = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_random]))
    value_green_used_random  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_random]))
    value_brown_used_random  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_random]))

    value_green_avail_emptier = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_emptier]))
    value_green_used_emptier  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_emptier]))
    value_brown_used_emptier  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_emptier]))

    value_green_avail_heuristic = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_heuristic]))
    value_green_used_heuristic  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_heuristic]))
    value_brown_used_heuristic  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_heuristic]))
    
    value_green_avail_sota = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_sota]))
    value_green_used_sota  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_sota]))
    value_brown_used_sota  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_sota]))

    value_green_avail_ce = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_ce]))
    value_green_used_ce  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_ce]))
    value_brown_used_ce  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_ce]))

    ##

    value_green_avail2 = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t2]))
    value_green_used2  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t2]))
    value_brown_used2 = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t2]))

    value_green_avail_random2 = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_random2]))
    value_green_used_random2  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_random2]))
    value_brown_used_random2  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_random2]))

    value_green_avail_emptier2 = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_emptier2]))
    value_green_used_emptier2 = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_emptier2]))
    value_brown_used_emptier2 = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_emptier2]))

    value_green_avail_heuristic2 = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_heuristic2]))
    value_green_used_heuristic2  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_heuristic2]))
    value_brown_used_heuristic2  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_heuristic2]))
    
    value_green_avail_sota2 = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_sota2]))
    value_green_used_sota2  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_sota2]))
    value_brown_used_sota2  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_sota2]))

    value_green_avail_ce2 = int(np.sum([np.sum(sublist) for sublist in green_energy_avail_t_ce2]))
    value_green_used_ce2  = int(np.sum([np.sum(sublist) for sublist in green_energy_used_t_ce2]))
    value_brown_used_ce2  = int(np.sum([np.sum(sublist) for sublist in brown_energy_used_t_ce2]))

    value_green_avail_solver     = [] if results_eval_solver1 != None else None
    value_green_used_solver      = [] if results_eval_solver1 != None else None
    value_brown_used_solver      = [] if results_eval_solver1 != None else None

    value_green_avail_solver2    = [] if results_eval_solver1 != None else None
    value_green_used_solver2     = [] if results_eval_solver1 != None else None
    value_brown_used_solver2     = [] if results_eval_solver1 != None else None

    if results_eval_solver1 != None:
       for n_eval in range(n_realizations):
            results_solver  = results_eval_solver1[n_eval]
            results_solver2  = results_eval_solver2[n_eval]

            if value_green_used_solver == [] and n_realizations == 1:
                value_green_avail_solver = results_solver['pw_nt_gr_sol'][TRANSIENT:]
                value_green_used_solver = getUsedGreenPW(results_solver)
                value_brown_used_solver = results_solver['pw_nt_br_sol'][TRANSIENT:]

                value_green_avail_solver2 = results_solver2['pw_nt_gr_sol'][TRANSIENT:]
                value_green_used_solver2 = getUsedGreenPW(results_solver2)
                value_brown_used_solver2 = results_solver2['pw_nt_br_sol'][TRANSIENT:]

            elif value_green_avail_solver == [] and n_realizations>1:
                value_green_avail_solver = [results_solver['pw_nt_gr_sol'][TRANSIENT:]]
                value_green_used_solver = [getUsedGreenPW(results_solver)]
                value_brown_used_solver = [results_solver['pw_nt_br_sol']][TRANSIENT:]

                value_green_avail_solver2 = [results_solver2['pw_nt_gr_sol'][TRANSIENT:]]
                value_green_used_solver2 = [getUsedGreenPW(results_solver2)]
                value_brown_used_solver2 = [results_solver2['pw_nt_br_sol']][TRANSIENT:]

            else:
                value_green_avail_solver.append(np.array(results_solver['pw_nt_gr_sol'])[:,TRANSIENT:])
                value_green_used_solver.append(getUsedGreenPW(results_solver))
                value_brown_used_solver.append(np.array(results_solver['pw_nt_br_sol'])[:,TRANSIENT:])

                value_green_avail_solver2.append(np.array(results_solver2['pw_nt_gr_sol'])[:,TRANSIENT:])
                value_green_used_solver2.append(getUsedGreenPW(results_solver2))
                value_brown_used_solver2.append(np.array(results_solver2['pw_nt_br_sol'])[:,TRANSIENT:])

       value_green_avail_solver = np.sum([np.sum(sublist) for sublist in value_green_avail_solver])
       value_green_used_solver = np.sum([np.sum(sublist) for sublist in value_green_used_solver])
       value_brown_used_solver = np.sum([np.sum(sublist) for sublist in value_brown_used_solver])

       value_green_avail_solver2 = np.sum([np.sum(sublist) for sublist in value_green_avail_solver2])
       value_green_used_solver2 = np.sum([np.sum(sublist) for sublist in value_green_used_solver2])
       value_brown_used_solver2 = np.sum([np.sum(sublist) for sublist in value_brown_used_solver2])

    color_vect = color_vect_noSolver

    if n_realizations >1:
        value_all_energy_policy     = value_green_avail + value_green_used + value_brown_used
        value_all_energy_heuristic  = value_green_avail_heuristic + value_green_used_heuristic + value_brown_used_heuristic
        value_all_energy_emptier    = value_green_avail_emptier + value_green_used_emptier + value_brown_used_emptier
        value_all_energy_random     = value_green_avail_random + value_green_used_random + value_brown_used_random
        value_all_energy_sota       = value_green_avail_sota + value_green_used_sota + value_brown_used_sota
        value_all_energy_ce         = value_green_avail_ce + value_green_used_ce + value_brown_used_ce

        value_all_energy_policy2    = value_green_avail2 + value_green_used2 + value_brown_used2
        value_all_energy_heuristic2 = value_green_avail_heuristic2 + value_green_used_heuristic2 + value_brown_used_heuristic2
        value_all_energy_emptier2   = value_green_avail_emptier2 + value_green_used_emptier2 + value_brown_used_emptier2
        value_all_energy_random2    = value_green_avail_random2 + value_green_used_random2 + value_brown_used_random2
        value_all_energy_sota2      = value_green_avail_sota2 + value_green_used_sota2 + value_brown_used_sota2
        value_all_energy_ce2        = value_green_avail_ce2 + value_green_used_ce2 + value_brown_used_ce2

        if results_eval_solver1 != None:
            value_all_energy_solver = value_green_avail_solver + value_green_used_solver + value_brown_used_solver

        if results_eval_solver2 != None:
            value_all_energy_solver2 = value_green_avail_solver2 + value_green_used_solver2 + value_brown_used_solver2

        norm_v = N_SERVERS*max_pw_server*n_realizations*(duration_realization-TRANSIENT)
        norm_v_solver1 = N_SERVERS*max_pw_server*n_realizations*(duration_realization_solver1-TRANSIENT)
        norm_v_solver2 = N_SERVERS * max_pw_server * n_realizations * (duration_realization_solver2 - TRANSIENT)

        green_energy_avail_j = (value_green_avail/norm_v, value_green_avail2/norm_v,
                                value_green_avail_heuristic/norm_v, value_green_avail_sota/norm_v, value_green_avail_ce/norm_v,
                                value_green_avail_random/norm_v, value_green_avail_emptier/norm_v,
                                value_green_avail_solver/norm_v_solver1, value_green_avail_solver2/norm_v_solver2)
        green_energy_used_j  = (value_green_used/norm_v, value_green_used2/norm_v,
                                value_green_used_heuristic/norm_v,  value_green_used_sota/norm_v, value_green_used_ce/norm_v,
                                value_green_used_random/norm_v,  value_green_used_emptier/norm_v,
                                value_green_used_solver/norm_v_solver1, value_green_used_solver2/norm_v_solver2)
        brown_energy_used_j  = (value_brown_used/norm_v, value_brown_used2/norm_v,
                                value_brown_used_heuristic/norm_v,  value_brown_used_sota/norm_v, value_brown_used_ce/norm_v,
                                value_brown_used_random/norm_v,  value_brown_used_emptier/norm_v,
                                value_brown_used_solver/norm_v_solver1, value_brown_used_solver2/norm_v_solver2)


        ##################

        fig, axsplt1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 4)
        fig.set_dpi(300)

        remainingGreenPW  = tuple([j1 - j2 for j1, j2 in zip(green_energy_used_j, green_energy_avail_j)])

        green_energy_used  = [j1 for j1 in green_energy_used_j]
        brown_energy_used  = [j1 for j1 in brown_energy_used_j]
        remainingGreenPWf  = [j1 for j1 in remainingGreenPW]

        legend_alg_b = ('GreenRL\n Fairness', 'GreenRL \n Profit', 'GreenH',
                   'CC23', 'CE25', 'Random', 
                   'Emptier', 'Optimal \n Fairness', 'Optimal \n Profit')
        
        y_pos = np.arange(len(legend_alg_b))
        plt.rcParams['text.usetex'] = True
        plt.rc('axes',  labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=MEDIUM_SIZE)
        plt.rc('ytick', labelsize=MEDIUM_SIZE)
        plt.rcParams['hatch.linewidth'] = 2.0  # Adjust the value as needed

        pg = plt.bar(y_pos, green_energy_used, color= color_vect[3], hatch = ['/', None,  None, None, None, None, None,'/', None], edgecolor = 'white', linewidth=0, align='center', alpha=0.7 )
        pb = plt.bar(y_pos, brown_energy_used, color= color_vect[0], hatch = ['/', None,  None, None, None, None, None,'/', None], edgecolor = 'white', linewidth=0, bottom=green_energy_used)
        pr = plt.bar(y_pos, remainingGreenPWf, color= color_vect[1], hatch = ['/', None,  None, None, None, None, None,'/', None], edgecolor = 'white', linewidth=0, alpha = 0.7)

        
        plt.legend((pg[0], pb[0], pr[0]), 
                   (r'Green Energy Consumed', 'Polluting Energy Consumed', 'Green Energy Remaining'), 
                   loc= 'lower left', bbox_to_anchor=(0.00, -0.03), ncol=3, fontsize="12", framealpha=0.30)
        plt.ylabel(ylabel)
        plt.xticks(y_pos, legend_alg_b)
        plt.axhline(y=0.0, color='k', linestyle='-', linewidth=0.5)
        plt.xlim((-0.5,8.5))

        plt.savefig(path_figure, bbox_inches='tight')
        # plt.show()

def plot_generic_total(results_eval, results_eval_heuristic, results_eval_random,
                       results_eval_emptier, results_eval_solver1, results_eval_sota1, results_eval_ce1,
                       results_eval2, results_eval_solver2, n_realizations, output_dir, print_figures, nameAttribute, yaxis_name,nameFig):

    ylabel  = yaxis_name
    name_attribute = nameAttribute

    path_figure = os.path.join('.', output_dir,   nameFig)

    data, data2                     = [],[]
    data_random,data_random2        = [],[]
    data_emptier,data_emptier2      = [],[]
    data_heuristic, data_heuristic2 = [],[]
    data_sota, data_sota2           = [],[]
    data_ce, data_ce2               = [],[]

    for n_time in range(n_realizations):
        results           = results_eval[n_time]
        results_random    = results_eval_random[n_time]
        results_emptier   = results_eval_emptier[n_time]
        results_heuristic = results_eval_heuristic[n_time]
        results_sota      = results_eval_sota1[n_time]
        results_ce        = results_eval_ce1[n_time]

        value           = results[name_attribute]
        value_random    = results_random[name_attribute]
        value_emptier   = results_emptier[name_attribute]
        value_heuristic = results_heuristic[name_attribute]
        value_sota      = results_sota[name_attribute]
        value_ce        = results_ce[name_attribute]
        
        if n_realizations == 1:
            data = value
            data_random    = value_random
            data_emptier   = value_emptier
            data_heuristic = value_heuristic
            data_sota      = value_sota
            data_ce        = value_ce
        else:
            data           = append_(data, value)
            data_random    = append_(data_random, value_random)
            data_emptier   = append_(data_emptier, value_emptier)
            data_heuristic = append_(data_heuristic, value_heuristic)
            data_sota      = append_(data_sota, value_sota)
            data_ce        = append_(data_ce, value_ce)

    for n_time in range(n_realizations):
        results2           = results_eval2[n_time]
        results_random2    = results_eval_random2[n_time]
        results_emptier2   = results_eval_emptier2[n_time]
        results_heuristic2 = results_eval_heuristic2[n_time]
        results_sota2      = results_eval_sota2[n_time]
        results_ce2        = results_eval_ce2[n_time]

        value2           = results2[name_attribute]
        value_random2   = results_random2[name_attribute]
        value_emptier2   = results_emptier2[name_attribute]
        value_heuristic2 = results_heuristic2[name_attribute]
        value_sota2      = results_sota2[name_attribute]
        value_ce2        = results_ce2[name_attribute]

        if n_realizations == 1:
            data2 = value2
            data_random2 = value_random2
            data_emptier2 = value_emptier2
            data_heuristic2 = value_heuristic2
            data_sota2      = value_sota2
            data_ce2        = value_ce2
        else:
            data2           = append_(data2, value2)
            data_random2    = append_(data_random2, value_random2)
            data_emptier2   = append_(data_emptier2, value_emptier2)
            data_heuristic2 = append_(data_heuristic2, value_heuristic2)
            data_sota2      = append_(data_sota2, value_sota2)
            data_ce2        = append_(data_ce2, value_ce2)

    data_solver1  = [] if results_eval_solver1 != None else None
    data_solver2  = [] if results_eval_solver2 != None else None

    if results_eval_solver1 != None:
       for n_time in range(n_realizations):
            results_solver1  = results_eval_solver1[n_time]
            results_solver2  = results_eval_solver2[n_time]
            value1 = results_solver1[name_attribute]
            value2 = results_solver2[name_attribute]
            if n_realizations == 1:
                data_solver1 = value1
                data_solver2 = value2
            else:
                data_solver1 = append_(data_solver1,value1)
                data_solver2 = append_(data_solver2,value2)

    color_vect = [color_vect_Solver[ii] for ii in [0,0,1,2,3,4,5,5,5]]

    if n_realizations == 1:
        pass
    else:
         plot_mean2(data, data_random, data_emptier, data_heuristic, data_solver1, data_sota, data_ce,
                 data2, data_solver2, n_realizations, confidence, color_vect, print_figures,
                 path_figure, ylabel,  singleList=1)

def plot_mean2(policy,  random,  emptier,  heuristic,  solver,  sota,  ce, 
               policy2, solver2, n_realizations, confidence, color_vect, print_figures, 
               path_figure, ylabel, singleList = 0):
    
        fig, plt1 = plt.subplots(1, 1)
        fig.set_size_inches(10, 4)
        fig.set_dpi(300)

        mean_policy    = np.average(policy)
        mean_random    = np.average(random)
        mean_emptier   = np.average(emptier)
        mean_heuristic = np.average(heuristic)
        mean_sota      = np.average(sota)
        mean_ce        = np.average(ce)

        std_policy     = np.std(policy)
        std_random     = np.std(random)
        std_emptier    = np.std(emptier)
        std_heuristic  = np.std(heuristic)
        std_sota       = np.std(sota)
        std_ce         = np.std(ce)

        mean_policy2    = np.average(policy2)
        std_policy2     = np.std(policy2)

        if singleList:
            dof  = n_realizations - 1
            dof1 = n_realizations
        else:
            dof = len(policy[0] * n_realizations) - 1
            dof1 = len(policy[0] * n_realizations)

        t_crit         = np.abs(t.ppf((1 - confidence) / 2, dof))
        conf           = (std_policy * t_crit / np.sqrt(dof1))     #* 2
        conf_random    = std_random * t_crit / np.sqrt(dof1)
        conf_emptier   = std_emptier * t_crit / np.sqrt(dof1)
        conf_heuristic = std_heuristic * t_crit / np.sqrt(dof1)
        conf_sota      = std_sota * t_crit / np.sqrt(dof1)
        conf_ce        = std_ce * t_crit / np.sqrt(dof1)
        conf2          = (std_policy2 * t_crit / np.sqrt(dof1))    #* 2

        mean_solver = np.average(solver)
        std_solver  = np.std(solver)
        conf_solver = std_solver * t_crit / np.sqrt(dof1)

        mean_solver2 = np.average(solver2)
        std_solver2  = np.std(solver2)
        conf_solver2 = std_solver2 * t_crit / np.sqrt(dof1)

        objects = ('GreenRL\n Fairness', 'GreenRL \n Profit', 'GreenH',
                   'CC23', 'CE25', 'Random', 
                   'Emptier', 'Optimal \n Fairness', 'Optimal \n Profit')
        yerr = (conf, conf2, conf_heuristic, conf_sota, conf_ce, conf_random, conf_emptier, conf_solver, conf_solver2)
        performance = [mean_policy, mean_policy2, mean_heuristic, mean_sota, mean_ce, mean_random, 
                       mean_emptier, mean_solver, mean_solver2]

        
        y_pos = np.arange(len(objects))
        plt.rcParams['text.usetex'] = True
        plt.rc('axes',   labelsize = MEDIUM_SIZE)
        plt.rc('xtick',  labelsize = MEDIUM_SIZE)
        plt.rc('ytick',  labelsize = MEDIUM_SIZE)
        plt.rc('legend', fontsize  = MEDIUM_SIZE)        
        plt.rcParams['hatch.linewidth'] = 2.0  # Adjust the value as needed
        plt.bar(y_pos, performance, yerr=yerr, color= color_vect,   
                hatch = ['/', None, None, None, None, None, None, '/', None], 
                edgecolor = 'white', align='center', alpha=0.7)
        plt.xticks(y_pos, objects)
        plt.ylabel(ylabel)
        plt.xlim((-0.5,8.5))
        fig.savefig(path_figure, bbox_inches='tight')
        if print_figures:
            plt.show()
        else:
            plt.close()


def plot_gCO2ekhw(results_eval, results_eval_heuristic, results_eval_random,
                       results_eval_emptier, results_eval_solver1, results_eval_sota1, results_eval_ce1,
                       results_eval2, results_eval_solver2, n_realizations, output_dir, print_figures, 
                       MAX_CYCLE, yaxis_name, nameFig):

    ylabel         = yaxis_name
    name_attribute = 'pw_nt_br_sol'
    max_pw_server  = MAX_CYCLE[0]
                                   
    path_figure = os.path.join('.', output_dir, nameFig)

    def get_gCO2perRealization(total_brown_power):
        polluting_percentOfTotalPw = np.sum(total_brown_power)/max_pw_server ## here 1 == One server full use 1 TS -> unit = 900 W * 10 s = 9 kWs
        TS_in_hours = 10/3600
        Watt_Server = 900
        total_kwh   = TS_in_hours * Watt_Server * polluting_percentOfTotalPw
        gco2e       = 372 * total_kwh

        return gco2e
        
    def get_gCO2perSessionOK(results_all_realz, interruptions = False):
        gCO2_Users  = []
        for n_real in range(n_realizations):
            thisRealization = results_all_realz[n_real]
            
            value_gCO2e = get_gCO2perRealization(thisRealization[name_attribute])

            numUsers_OK = np.sum(thisRealization['accepted_jobs_t']) 
            if interruptions:
                j_interrup = np.sum(thisRealization['interruptedpower_t']) \
                            + np.sum(thisRealization['interrupted_rejected_t']) \
                            + np.sum(thisRealization['interruptedelay_t'])    
                numUsers_OK = numUsers_OK - j_interrup    
               
            value_gCO2e_usersOK = value_gCO2e/numUsers_OK

            gCO2_Users = append_(gCO2_Users, value_gCO2e_usersOK)

        return gCO2_Users
           
    data           = get_gCO2perSessionOK(results_eval, interruptions = True)
    data_random    = get_gCO2perSessionOK(results_eval_random)
    data_emptier   = get_gCO2perSessionOK(results_eval_emptier)
    data_heuristic = get_gCO2perSessionOK(results_eval_heuristic, interruptions = True)
    data_sota      = get_gCO2perSessionOK(results_eval_sota1)
    data_ce        = get_gCO2perSessionOK(results_eval_ce1)
        
    data2           = get_gCO2perSessionOK(results_eval2, interruptions = True)
      
    data_solver1  = [] if results_eval_solver1 != None else None
    data_solver2  = [] if results_eval_solver2 != None else None

    data_solver1 = get_gCO2perSessionOK(results_eval_solver1)
    data_solver2 = get_gCO2perSessionOK(results_eval_solver2)

    color_vect = [color_vect_Solver[ii] for ii in [0,0,1,2,3,4,5,5,5]]
    
    plot_mean2(data, data_random, data_emptier, data_heuristic, data_solver1, data_sota, data_ce,
            data2, data_solver2, n_realizations, confidence, color_vect, print_figures,
            path_figure, ylabel, singleList=1)
    
#############################################################################
#%%##########################################################################
#############################################################################
#############################################################################

experiment_dirs1      = os.path.join(exp_folder, experiment_dir1)
experiment_dirs2      = os.path.join(exp_folder, experiment_dir2)
outputFig_path        = os.path.join(output_folder, name_exp)
if not os.path.exists(outputFig_path):
    os.makedirs(outputFig_path, exist_ok=True)
    
results_eval_file1   = os.path.join(experiment_dirs1, 'results_eval.pkl')
results_solver_file1 = os.path.join(experiment_dirs1, 'results_solver.pkl')

results_eval_file2   = os.path.join(experiment_dirs2, 'results_eval.pkl')
results_solver_file2 = os.path.join(experiment_dirs2, 'results_solver.pkl')

### READING FILES ##########

with open(results_eval_file1, 'rb') as mfile:
    results_eval1           = pickle.load(mfile)
    env1                    = pickle.load(mfile)
    results_eval_random1    = pickle.load(mfile)
    results_eval_emptier1   = pickle.load(mfile)
    results_eval_heuristic1 = pickle.load(mfile)
    results_eval_sota1      = pickle.load(mfile)
    results_eval_ce1        = pickle.load(mfile)

with open(results_solver_file1, 'rb') as mfile:
    results_solver1 = pickle.load(mfile)

with open(results_eval_file2, 'rb') as mfile:
    results_eval2           = pickle.load(mfile)
    env2                    = pickle.load(mfile)
    results_eval_random2    = pickle.load(mfile)
    results_eval_emptier2   = pickle.load(mfile)
    results_eval_heuristic2 = pickle.load(mfile)
    results_eval_sota2      = pickle.load(mfile)
    results_eval_ce2        = pickle.load(mfile)

with open(results_solver_file2, 'rb') as mfile:
    results_solver2 = pickle.load(mfile)
    
N_SERVERS = len(results_eval1[0]['pw_nt_sol'])
MAX_CYCLE = [MAX_CYCLE_NORM_SERVER]*N_SERVERS

#%%### actual plotting for objetive
n_realizations_eval = len(results_eval1)
tot_time_evaluation = len(results_eval1[0]["job_pc_list"])
TRANSIENT           = int(TRANSIENT_PERC*tot_time_evaluation)

plt.rcParams['text.usetex'] = True
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)

#%%
namefig = f"n{N_SERVERS}_operatorCost.{format_fig}"

plot_generic_total(results_eval1, results_eval_heuristic1, results_eval_random1,
                    results_eval_emptier1, results_solver1, results_eval_sota1, results_eval_ce1,
                    results_eval2, results_solver2, n_realizations_eval,
                    outputFig_path, PRINT_FIGURES, 'costPWTotal',
                    r'Operator\'s Energy Cost ($\delta P$)', namefig)

namefig = f'n{N_SERVERS}_profitMargin' + '.' + format_fig

plot_generic_total(results_eval1, results_eval_heuristic1, results_eval_random1,
                    results_eval_emptier1, results_solver1, results_eval_sota1, results_eval_ce1,
                    results_eval2, results_solver2, n_realizations_eval,  outputFig_path, PRINT_FIGURES,
                    'Objective_function_rev_minus_cost', r' Profit margin  ($\bar{B}$)', namefig)

######## FAIRNESS  #############

namefig = f"n{N_SERVERS}_fairnessObj.{format_fig}"


dict_values1 = getDataAcceptedJobs(results_eval1, results_eval_heuristic1, results_eval_sota1, results_eval_ce1, results_eval_random1, results_eval_emptier1, results_solver1, n_realizations_eval)

dict_values2 = getDataAcceptedJobs(results_eval2, results_eval_heuristic2, results_eval_sota2, results_eval_ce2, results_eval_random2, results_eval_emptier2, results_solver2, n_realizations_eval)

namefig = f'n{N_SERVERS}_userAcceptance' + '.' + format_fig

plot_jobs_accepted_rejected_all(dict_values1, dict_values2, outputFig_path,
                                r"Percentage of users", namefig)

#%%

namefig = f'n{N_SERVERS}_pwStats' + '.' + format_fig

plot_pw_alg(results_eval1, results_eval_heuristic1, results_eval_random1,
                    results_eval_emptier1, results_solver1, results_eval_sota1, results_eval_ce1,
                    results_eval2, results_eval_heuristic2, results_eval_random2,
                    results_eval_emptier2, results_solver2, results_eval_sota2, results_eval_ce2, n_realizations_eval, TRANSIENT,
                    outputFig_path, N_SERVERS, MAX_CYCLE,
                    r"$\%$ of total MEC power", namefig)

#%%

namefig = f'n{N_SERVERS}_gCO2e_statistics' + '.' + format_fig

plot_gCO2ekhw(results_eval1, results_eval_heuristic1, results_eval_random1,
            results_eval_emptier1, results_solver1, results_eval_sota1, results_eval_ce1,
            results_eval2, results_solver2, n_realizations_eval,
            outputFig_path, PRINT_FIGURES, MAX_CYCLE,
            r'gCO$_{2}$e/completed MAR session', namefig)