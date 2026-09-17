
##############################################################################
#%% Package imports ##########################################################

import os, io, sys, time
import numpy as np
from collections import Counter
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

import pyscipopt
from pyscipopt import Model, quickprod, quicksum
from pyscipopt import log as sciplog
from pyscipopt import exp as scipexp
from itertools import chain

from auxiliarFunctions import PrintSwitch, TimeCounter, checkFile_and_rename, append_

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})

########### Class and method definitions

# IPOPT=true
# WORHP=true
# FILTERSQP=true

class Solver():
    """
        Exact/Heuristic Optimization Solver Wrapper based on SCIP.
        Employs a Receding Horizon (Sliding Window) approach to solve job allocation,
        migration, and power dispatching optimization problems over a multi-timeslot horizon.
    """
    markerList = ["o", "v", "x","s", "D", ".", ", ", "^", "<", ">", "8", "p", "P",
                  "*", "h", "H", "+", "X", "d", "/", "_", "1", "2", "3", "4"]

    def __init__(self, env, time_opt, T_tot, duration_cost_migration = 1, margin_last_jobs = None, deadline_solver  = 10,
                 num_Max_iter = 5, gapPostdeadline  = .02, gapIncrementStop = 0, revcost_function=1, nologFair = 0):
        """
                Initializes the Solver with environment parameters, optimization horizon settings,
                and solver termination bounds.

                :param env: The simulation environment instance (WorldEnv) containing system capacities.
                :param time_opt: Optimization time window (T) for look-ahead planning.
                :param T_tot: Total duration (timeslots) of the evaluation episode.
                :param duration_cost_migration: Migration duration in timeslots.
                :param margin_last_jobs: Extension window beyond T to allow ongoing jobs to complete.
                :param deadline_solver: Maximum time (seconds) allowed per optimization iteration.
                :param num_Max_iter: Maximum recursive solver extension iterations if timeout occurs.
                :param gapPostdeadline: Target optimality gap target (e.g. 0.02 = 2%).
                :param gapIncrementStop: Minimum gap improvement threshold to continue iterations.
                :param revcost_function: 1 for profit formulation (Revenue - Energy Cost), 0 for fairness formulation.
                :param nologFair: 1 to omit log in fairness objective, 0 to include log.
        """
        self.revcost_function = revcost_function
        self.nologInFairness  = nologFair
        
        ###### Class inputs ##############
        self.deadline_solver  = deadline_solver # Max. time for each opt. iteration to finish
        self.num_Max_iter     = num_Max_iter    # Max. number of iterations of optimization
        self.gapPostdeadline  = gapPostdeadline  ## % Gap w.r.t. optimal for next interation stop
        self.gapIncrementStop = gapIncrementStop ## keep it 0 for now. It can give bad gap solutions

        self.T_tot    = T_tot # total number of timeslots in simulation
        self.T        = time_opt # Time window for which the system is optimized
        self.N        = env.N_SERVERS

        # Extended margin to allow jobs near the boundary to finish execution
        self.margin_end = env.JobSessionLength if margin_last_jobs == None else margin_last_jobs
        self.T_ext    = self.T + self.margin_end

        # Power mapping hyper-parameters (alpha * PC + gamma) and migration penalty (beta)
        self.alpha, self.beta, self.gamma  = env.alpha, env.beta, env.gamma
        self.rho_p, self.rho_d, self.rho_r = env.rho_p, env.rho_d, env.rho_r

        self.C_max_n        = env.MAX_PC_SERVERS # this is a vector now
        self.max_power_node = env.MAX_PW_SERVERS

        ##### SYSTEM PARAMETERS
        self.c_j_min = env.c_jn # minimum amount of processing per job per time
        self.mig_dur = duration_cost_migration # Duration of migrations

        self.revenue  = env.REV_FACTOR
        self.cost_pw  = env.COST_FACTOR
        
        ##### DELAY PARAMETERS
        # Calculates max allowed 1-way delay derived from deadline minus fixed switching/transmission & computation delays.
        # Divided by 2 to account for Round-Trip Time (RTT).
        self.MAX_1WAY_DELAY = (env.DEADLINE_DELAY - 2*env.TRASMISSION_DELAY - env.COMPUTATION_DELAY)/2

        ##### Solver State Dictionary #####
        # Keeps track of active jobs carried over between consecutive sliding window iterations.
        self.state = {'C_j_in':      [], 'previousPCs': [],  'remainingTSs': [],
                      'currentNodes':[], 'rev_j_t_in':  [],  'wdlay_in':     []}

        ### Performance Tracking Variables ###
        self.pcAllocated   = []
        self.delayFinished = []
        self.brownPw       = []
        self.totalPw       = []
        self.revArrived    = []
        self.revAccepted   = []
        self.acceptedJobs  = []
        self.arrivedJobs   = []
        self.migration_y   = []
        self.acceptanceX   = []

    def createInstance(self, jobs, jobs_dur, jobs_wlessDelay, nodes_propDelay, greenPW, state = None):
        """
                Instantiates an OptimizationInstance class corresponding to a single sliding window step.
        """
        theState = state if state != None else self.state
        return Solver.OptimizationInstance(self, jobs, jobs_dur, jobs_wlessDelay, nodes_propDelay, greenPW, theState)

    def round2int(array):
        """Helper to round float arrays from SCIP solution variables into integers."""

        return np.int_(np.round(array))
    def scipmin(a,b):
        """Formulates minimum operation using SCIP algebraic expressions: min(a,b) = (a+b - |a-b|)/2."""

        return (a+b - abs(a-b))/2
    def scipmax(a,b):
        """Formulates maximum operation using SCIP algebraic expressions: max(a,b) = (a+b + |a-b|)/2."""

        return (a+b + abs(a-b))/2

    def checkStatusSolution(status, p = PrintSwitch(True)):
        """Logs and returns string tags based on SCIP termination status."""

        l_start = '\n -------------------\n     !!!!!!!!!        '
        l_end   = '\n     !!!!!!!!!        \n -------------------\n'
        if status == 'optimal':
            p.print_(l_start + '\n Optimal solution!' + l_end)
            text_save = "Optimal"
        elif status ==  'infeasible':
            p.print_(l_start + '\n Problem is not feasible!' + l_end)
            text_save = "Unfeasible"
        elif status == 'timelimit':
            p.print_(l_start + '\n Time Limit reached!!' + l_end)
            text_save = "Time_limit"
        elif status == 'gaplimit':
            p.print_(l_start + '\n Gap Limit reached!!' + l_end)
            text_save = "Gap_reached"
        else:
            p.print_(l_start, '\n Current Status : '+ status + l_end)
            text_save = "Not_optimal"
        return text_save

    def getValVars(model, x_matrix):
        """Extracts numerical values of SCIP decision variables after model optimization."""
        getSingleVar = lambda x: model.getVal(x)
        vectorized_func = np.vectorize(getSingleVar)
        return vectorized_func(x_matrix)

    def optimize(self, model, doPrint, iteration, prevGap, gapDiffStop):
        """
                Triggers the SCIP optimization process. Implements recursive solver extensions
                if time limits are hit without reaching the target optimality gap.

                :param model: The PySCIPOpt Model instance.
                :param doPrint: Verbosity flag.
                :param iteration: Current extension attempt count.
                :param prevGap: Gap recorded in the preceding attempt.
                :param gapDiffStop: Threshold of gap reduction required to justify extending execution.
                :return: (optimized_model, status)
        """
        doPrint=1
        ## Solving the problem
        if doPrint:
            model.optimize()
        else:
            trap = io.StringIO()
            with redirect_stdout(trap):
                model.optimize() #trap.getvalue()

        # Check solver termination status
        probStatus = model.getStatus()

        if probStatus == 'infeasible': ### No solution. This should not happen
            pass
        elif probStatus == 'optimal': ### Done
            pass
        elif probStatus == 'gaplimit': ### No more iterations
            print(f'\n+-+-+-+-+-+-+-+-+\n Gap bound of {self.gapPostdeadline} reached in iteration {iteration} - Gap: {model.getGap():.2f}\n+-+-+-+-+-+-+-+-+')
            pass
        elif probStatus == 'timelimit': ### Loose gap requirement
            # conditions could be modified based on the iteration
            currentGap  = model.getGap()
            print(f'\n\n ----------- Iteration: {iteration} - Gap: {currentGap}')
            if (prevGap - currentGap < gapDiffStop):
                probStatus == ' gapIncrementlimit'
                print(f'\n+-+-+-+-+-+-+-+-+\n Gap increment of {gapDiffStop} reached \
                      in iteration {iteration} - Gap increment: {prevGap - currentGap:.2f}.\
                      Improvement too slow\n+-+-+-+-+-+-+-+-+')
            elif iteration < self.num_Max_iter:
                newDeadline = self.deadline_solver*(iteration+2)
                model.setParam('limits/time', newDeadline)
                model.setParam('limits/gap', self.gapPostdeadline)
                iteration += 1
                model, probStatus = self.optimize(model, doPrint, iteration, currentGap, gapDiffStop)
        else:
            pass

        return model, probStatus


    def createPlottingStats(self, greenPW, cost_factor, window_size):
        """
                Compiles performance metrics (revenues, brown energy usage, fairness, objective values)
                across rolling windows and for the entire simulation realization.

                :param greenPW: Array of available green power per node per timeslot.
                :param cost_factor: Unit cost of brown power.
                :param window_size: Size of sliding evaluation window.
                :return: Dict containing compiled summary results.
        """

        revAcc_movAver  = np.zeros(self.T_tot-window_size+1)
        br_pw_movAver   = np.zeros(self.T_tot-window_size+1)
        delay_movAver   = np.zeros(self.T_tot-window_size+1)
        revTot_movAver  = np.zeros(self.T_tot-window_size+1)
        greenPW_movAver = np.zeros(self.T_tot-window_size+1)
        greenPW_np =  np.array(greenPW)

        ##### moving average results ####################

        for tt in range(self.T_tot-window_size+1):
            br_pw_movAver[tt] = np.sum(self.brownPw[tt : tt + window_size])

            delay_movAver[tt] =  1 #- self.rho_d * round( mean_delay_window/ (self.D_max), 2)

            revAcc_movAver[tt]  = np.sum(self.revAccepted[tt : tt + window_size])
            revTot_movAver[tt]  = np.sum(self.revArrived[tt : tt + window_size])
            greenPW_movAver[tt] = np.sum(greenPW_np[:,tt : tt + window_size])
        results_solver = {}

        obj_rev_movAver = revAcc_movAver/revTot_movAver
        obj_rev_movAver = [round(val,2) if not (np.isnan(val) or np.isinf(val)) else 0 for val in obj_rev_movAver]
        obj_rev_movAver_factor = np.array([self.rho_r*val + (1-self.rho_r) for val in obj_rev_movAver])

        max_brownPW_movAver = np.sum(self.max_power_node)*window_size - greenPW_movAver
        obj_pw_movAver = 1 - self.rho_p* br_pw_movAver/max_brownPW_movAver
        obj_pw_movAver = [obj_pw_movAver[nn] if max_brownPW_movAver[nn] != 0 else 1 for nn in range(len(obj_pw_movAver))]

        obj_rev_profit = (revAcc_movAver - cost_factor*br_pw_movAver)/revTot_movAver
        obj_rev_profit = [obj_rev_profit[nn] if revTot_movAver[nn] != 0 else 1 for nn in range(len(obj_rev_profit))]

        ##### total time results ####################

        objFair_rev = self.rho_r*np.sum(self.revAccepted)/np.sum(self.revArrived) + (1-self.rho_r)
        max_brownPW  = np.sum(self.max_power_node)* self.T_tot - np.sum(greenPW)
        if max_brownPW == 0:
            objFair_pw = 1
        else:
            objFair_pw =1 - self.rho_p*np.sum(self.brownPw)/max_brownPW

        objFairnoLog = objFair_rev*objFair_pw
        objFairnoLog_movAver = obj_rev_movAver_factor * obj_pw_movAver
        results_solver['revRat_movingAv']        = obj_rev_movAver
        results_solver['revFairness_movingAv']   = obj_rev_movAver_factor
        results_solver['powerFairness_movingAV'] = obj_pw_movAver
        results_solver['objFairness_movingAv_nolog'] = objFairnoLog_movAver
        results_solver['objFairness_movingAv_log']   = np.log(objFairnoLog_movAver)
        results_solver['objFairness_movingAv']       = objFairnoLog_movAver if self.nologInFairness else np.log(objFairnoLog_movAver)
        results_solver['delay_moving_average']   = delay_movAver

        results_solver['revenue_movingAV']       = revAcc_movAver
        results_solver['revenue_cost_movingAV']  = revAcc_movAver - cost_factor*br_pw_movAver
        results_solver['obj_function_revenue_costMV'] = obj_rev_profit

        results_solver['pw_nt_sol']    = np.array([[ts[ii] for ts in self.totalPw] for ii in range(self.N)])
        results_solver['pw_nt_br_sol'] = np.array([[ts[ii] for ts in self.brownPw] for ii in range(self.N)])
        results_solver['total_brown_used'] = np.sum(np.array([[ts[ii] for ts in self.brownPw] for ii in range(self.N)]))
        results_solver['total_green_available'] = np.sum(greenPW)
        results_solver['total_green_not_used'] = np.sum(greenPW) - (np.sum(np.array([[ts[ii] for ts in self.totalPw] for ii in range(self.N)])) - np.sum(np.sum(np.array([[ts[ii] for ts in self.brownPw] for ii in range(self.N)]))))

        results_solver['pw_nt_gr_sol'] = greenPW
        results_solver['cost_br_pw']   = np.sum(results_solver['pw_nt_br_sol'],axis=0)*cost_factor

        results_solver['accepted_jobs_t'] = self.acceptedJobs
        results_solver['rejected_jobs_t'] = np.array(self.arrivedJobs) - np.array(self.acceptedJobs)

        #     results_solver['job_list'] = job_list
        results_solver['RevenueRatioTotal'] = np.sum(self.revAccepted)/np.sum(self.revArrived)
        results_solver['Objective_function_rev_minus_cost'] = (np.sum(self.revAccepted)-cost_factor*round(np.sum(self.brownPw),2))/np.sum(self.revArrived)
        results_solver['revenueTotal'] = np.sum(self.revAccepted)
        results_solver['costPWTotal']  = cost_factor*round(np.sum(self.brownPw),2)

        results_solver['Objective_function_fairness_nolog'] = objFairnoLog
        results_solver['Objective_function_fairness_log']   = np.log(objFairnoLog)
        results_solver['objFunction_Fairness']    = objFairnoLog if self.nologInFairness else np.log(objFairnoLog)
        results_solver['normPower_fair']  = objFair_pw
        results_solver['normRev_fair']    = objFair_rev

        print('objFair_pw',objFair_pw)
        print('objFair_rev',objFair_rev)
        print('objFairnoLog',objFairnoLog)
        print('objFair_pw*objFair_rev',objFair_pw*objFair_rev)
        print("results_solver['objFunction_Fairness']",results_solver['objFunction_Fairness'] )

        return results_solver

    ###############################################################################
    # OPTIMIZATION INSTANCE CLASS
    ###############################################################################

    class OptimizationInstance():
        """
                Represents a single optimization problem over a sliding time window T_ext.
                Constructs PySCIPOpt variables, linear constraints, and objectives.
        """
        def __init__(self, solver, joblist, duration_jobs, wlessDelay_jobs, propDelay_nodes, green_PW, currentState):
            self.parent  = solver

            # Flatten inputs across timeslots for the window
            self.C_j        = [job for job_t in joblist for job in job_t]
            self.dur_j      = [dur for dur_ts in duration_jobs for dur in dur_ts]
            self.wlessDel_j = [wd_job for wd_job_t in wlessDelay_jobs for wd_job in wd_job_t]
            self.propgDel_n = propDelay_nodes
            
            self.propgDel_max = np.max(np.matrix(propDelay_nodes))
                        
            self.arriv_t = [idx for idx, job_t in enumerate(joblist) for job in job_t]
            self.arriv_t.sort()

            # Potential revenue per job
            self.rev_j_t = [self.parent.revenue*self.C_j[jj] for jj in range(len(self.C_j))]
            self.rev_j   = [self.rev_j_t[jj]*self.dur_j[jj] for jj in range(len(self.C_j))]
            self.J = len(self.C_j)

            # Extend green power availability profile across the safety margin end
            self.greenPW = np.array([np.append(gE_node_ii, [gE_node_ii[-1]]*self.parent.margin_end)  for gE_node_ii in green_PW])

            if len(joblist) != self.parent.T:
                print(len(joblist), self.parent.T)
                sys.exit(f"Input time interval is different than T = {self.parent.T}, job_list = {len(joblist)}")

            # Active jobs carried over from previous window
            self.C_j_in       = currentState['C_j_in']
            self.remainingTSs = currentState['remainingTSs']
            self.previousPCs  = currentState['previousPCs']
            self.currentNodes = currentState['currentNodes']
            self.rev_j_t_in   = currentState['rev_j_t_in']
            self.wdlay_in     = currentState['wdlay_in']
            self.J_in         = len(self.C_j_in)

            self.rev_j_t_all = self.rev_j_t + self.rev_j_t_in #• concatelate lists

        def updateHistoryList(self, oldList, newValue):
            return append_(oldList, newValue)

        def solveInstance(self, talkative, nolog=False):
            """
            Builds and solves the Mixed Integer Linear Program (MILP) model in PySCIPOpt.
            """

            if self.J + self.J_in == 0:
                sol = {'arriv_t': [], 'solVal':None, 'objR': None,
                        'objP':None, 'objD':None, 'gap':None, 'x':[],
                        'c': [], 'y':[], 'a_j': [], 'sol':None}
                self.parent.state = {'C_j_in':  [],
                                'remainingTSs': [],
                                'previousPCs':  [],
                                'currentNodes': [],
                                'rev_j_t_in':   [],
                                'wdlay_in':     []}
                self.parent.pcAllocated   = self.updateHistoryList(self.parent.pcAllocated,   np.zeros(self.parent.N))
                self.parent.delayFinished = self.updateHistoryList(self.parent.delayFinished, [])
                self.parent.brownPw       = self.updateHistoryList(self.parent.brownPw,       np.zeros(self.parent.N))
                self.parent.totalPw       = self.updateHistoryList(self.parent.totalPw,       np.zeros(self.parent.N))
                self.parent.revArrived    = self.updateHistoryList(self.parent.revArrived,    0)
                self.parent.revAccepted   = self.updateHistoryList(self.parent.revAccepted,   0)
                self.parent.acceptedJobs  = self.updateHistoryList(self.parent.acceptedJobs,  0)
                self.parent.arrivedJobs   = self.updateHistoryList(self.parent.arrivedJobs,   0)
                self.parent.migration_y   = self.updateHistoryList(self.parent.migration_y,   0)
                self.parent.acceptanceX   = self.updateHistoryList(self.parent.acceptanceX,   0)

                return (None, sol, self.parent.state)
            else:
                p  = PrintSwitch(talkative)
                tc = TimeCounter(talkative)
                p.print_('\n Parameters set. Now defining the model\n')
                tc.setTimeOn()

                self.model = Model('solver_opt')
                self.model.setParam('limits/time', self.parent.deadline_solver)
                self.model.setParam('limits/gap', 0)

                ### Define variables x and c
                x = [[[self.model.addVar(lb=0, ub=1, vtype='B', name=f'x_{j+1}_{n+1}_{t+1}')
                           for t in range(self.parent.T_ext)] for n in range(self.parent.N)] for j in range(self.J_in + self.J)]

                c = [[[self.C_j[j]]*self.parent.T_ext]*self.parent.N for j in range(self.J)] \
                    + [[[self.C_j_in[j]]*self.parent.T_ext]*self.parent.N for j in range(self.J_in)] # now is not a variable...

                tc.offAndOn()

                ### numpy version of variables
                x_np = np.array(x)
                c_np = np.array(c)

                self.given_C = c_np

                rev_j_in_Text  =  [self.rev_j_t_in[jj]* min(self.parent.T_ext, self.remainingTSs[jj]) for jj in range(self.J_in)]
                rev_j_new_Text =  [self.rev_j_t[jj]   * min(self.parent.T_ext, self.dur_j[jj])        for jj in range(self.J)   ]
                self.revenue_currentJobs = rev_j_new_Text + rev_j_in_Text #concatenation


                p.print_('    1 - Variables defined')
                tc.getTimeOff(text='    ')
                ### ------------------------------------------------------------------#
                ### ----- OBJ FUNCTION -------
                ### ------------------------------------------------------------------#

                ##% Creating REVENUE part for objective  %%%%%%%%%%%%%%%%%%%%%%%%%%
                p.print_('\n    2 - Defining revenues')
                p.print_('       2a - Defining "acceptance" of each job')
                ### In order to implement the acceptance ratio, since the product of all x's takes
                ### exponential time from N = 4, T = 4..., we create an auxiliar variable for each
                ### of the jobs. This variable is set to be the OR function of the binary set
                ### {x_jnt for all n,t}, such that var = 0 if all(x) = 0 (job rejected)
                ### and var = 1 if exists at least an x = 1
                tc.setTimeOn()
                ### revenue ratio only for new jobs
                a_j = [self.model.addVar(vtype='B', name=f'a_j_var{jj}') for jj in range(self.J + self.J_in)]

                for jj in range(self.J + self.J_in):
                    self.model.addConsOr(list(chain(*x[jj])), a_j[jj])
                tc.offAndOn(text = '       ')

                ### Revenue per accepted job
                p.print_('\n       2b - Defining Revenue array per accepted job')
                # actual_rev = np.inner(self.revenue_j, a_j[:self.J])
                actual_rev = np.inner(self.revenue_currentJobs, a_j)
                tc.offAndOn(text = '       ')

                ### Total revenue of arrived (accepted and rejected) jobs
                p.print_('\n       2c - Getting max. total revenue')
                total_revenue = np.sum(self.revenue_currentJobs)
                tc.offAndOn(text = '       ')

                ### Ratio actual revenue vs maximum revenue
                revRatio = actual_rev/total_revenue if total_revenue != 0 else 1
                p.print_('\n       2d - Getting Ratio actual revenue vs maximum revenue')
                OBJ_R = revRatio * self.parent.rho_r + (1 - self.parent.rho_r)
                tc.getTimeOff(text = '       ')

                #%% Creating POWER part for objective       %%%%%%%%%%%%%%%%%%%%%%%%%%
                p.print_('\n    3 - Defining power objective')
                tc.setTimeOn()

                ##### Defining migration indicator #########################
                y = [[[self.model.addVar(lb=0, ub=1, vtype='B', name=f'y_{j+1}_{n+1}_{t+1}')
                           for t in range(self.parent.T_ext)] for n in range(self.parent.N)] for j in range(self.J_in + self.J)]
                y_np = np.array(y) ### numpy version of variables

                x_past =  [[0]*self.parent.N]*(self.J_in) # initalizing to save currentNodes in same format as x
                for jj in range(self.J_in): ## allocating previous node for migrations
                    if  self.currentNodes[jj] != None:
                        x_past[jj][self.currentNodes[jj]] = 1

                for jj in range(self.J + self.J_in):
                    for nn in range(self.parent.N):
                        for tt in range(1,self.parent.T_ext): # check if migration or start
                            self.model.addCons(y[jj][nn][tt] == (x[jj][nn][tt] - x[jj][nn][tt-1])*x[jj][nn][tt])

                        ### handling first time separately
                        if jj >= self.J: # if old job
                            self.model.addCons(y[jj][nn][0] == (x[jj][nn][0] - x_past[jj-self.J][nn])*x[jj][nn][0])
                        else: # new job
                            if self.arriv_t[jj] == 0: # was it accepted?
                                self.model.addCons(y[jj][nn][0] == x[jj][nn][0])
                            else: # no cost 'cause doesn't exist
                                self.model.addCons(y[jj][nn][0] == 0)


                ### Getting power from proc. cycles runned and migrations executed
                power_jnt = (self.parent.alpha*c_np + self.parent.gamma)*x_np + self.parent.beta*y_np

                ##### Defining brown energy consumed:
                power_nt_total    = np.sum(power_jnt, axis=0)   # Power consumed per node and per time
                diff_pw_tot_green = power_nt_total - self.greenPW # Difference between pw used and green pw available

                ## To adapt to PySCIPOpt:  define brown pw at node as a variable with lower bound = 0
                ## Then, add constraint (brown power at node n at time t) = maximum(0, diff_pw_tot_green)
                ## to avoid negative values if green energy > power used. We use max(x,0) = (x+abs(x))/2
                power_nt_brown = [[self.model.addVar(vtype='C', name=f'pw_brown_aux_var_{nn}_{tt}', lb = 0)
                                                       for tt in range(self.parent.T_ext)] for nn in range(self.parent.N) ]
                [[self.model.addCons(power_nt_brown[nn][tt] == Solver.scipmax(diff_pw_tot_green[nn,tt], 0)) # diff_pw_tot_green+ abs(diff_pw_tot_green[nn,tt]))/2)
                                                       for tt in range(self.parent.T_ext)] for nn in range(self.parent.N) ]
                ### total power over nodes and time
                P_brown_used = quicksum(list(chain(*power_nt_brown)))
                P_tot_used   = quicksum(list(chain(*power_nt_total)))

                ### Objective function term for power
                max_brownPW = np.sum(self.parent.max_power_node)*self.parent.T_ext - np.sum(self.greenPW)
                #OBJ_P  = 1 - self.parent.rho_p*P_brown_used/(self.parent.max_power_node*self.parent.N*self.parent.T_ext)
                if max_brownPW == 0:
                    OBJ_P = 1
                else:
                    OBJ_P = 1 - self.parent.rho_p * P_brown_used / max_brownPW

                #print('OBJ_P',OBJ_P)
                tc.getTimeOff(text = '       ')

                #######################################################################
                #%% ------ CONSTRAINTS ------ #########################################
                #######################################################################
                p.print_('    5 - Add constraints')
                tc.setTimeOn()

                ##### JOBS ALREADY IN THE SYSTEM MUST BE SERVED ####################

                for jj in range(self.J, self.J + self.J_in):
                    self.model.addCons(a_j[jj] >= 1)

                ##### ALLOCATION constraint  #######################################

                for jj in range(self.J + self.J_in):
                    for tt in range(self.parent.T_ext):
                        self.model.addCons(quicksum(x[jj][nn][tt] for nn in range(self.parent.N)) <= 1)

                ##### Proc. cycles allocated constraint  ###########################
                ##### In lower and upper bounds of variables

                ##### CPU node constraint  #########################################

                for nn in range(self.parent.N):
                    for tt in range(self.parent.T_ext):
                        self.model.addCons(quicksum(x[jj][nn][tt]*c[jj][nn][tt] for jj in range(self.J + self.J_in)) <= self.parent.C_max_n[nn])

                ##### Causal time constraint: Not served when job doesnot exist ####
                ## for new jobs
                for jj in range(self.J):
                    for tt in range(self.arriv_t[jj]):
                        self.model.addCons(quicksum(x[jj][nn][tt] for nn in range(self.parent.N)) == 0)

                    for tt in range(self.arriv_t[jj]+self.dur_j[jj], self.parent.T_ext):
                        self.model.addCons(quicksum(x[jj][nn][tt] for nn in range(self.parent.N)) == 0)

                ##  for jobs in system
                for jj in range(self.J, self.J + self.J_in):
                    for tt in range(self.remainingTSs[jj - self.J], self.parent.T_ext):
                        self.model.addCons(quicksum(x[jj][nn][tt] for nn in range(self.parent.N)) == 0)

                ##### Causal time constraint: Served all the time if accepted ######
                ## for new jobs
                for jj in range(self.J):
                    for tt in range(self.arriv_t[jj]+1, min(self.parent.T_ext, self.arriv_t[jj]+self.dur_j[jj])):
                            # print(tt, self.arriv_t[jj]+1, self.parent.T_ext, self.arriv_t[jj]+self.dur_j[jj], min(self.parent.T_ext, self.arriv_t[jj]+self.dur_j[jj]))
                            self.model.addCons(quicksum(x[jj][nn][tt] for nn in range(self.parent.N)) == quicksum(x[jj][nn][self.arriv_t[jj]] for nn in range(self.parent.N)))

                ##  for jobs in system
                for jj in range(self.J, self.J + self.J_in):
                    for tt in range(min(self.parent.T_ext, self.remainingTSs[jj - self.J])):
                        self.model.addCons(quicksum(x[jj][nn][tt] for nn in range(self.parent.N)) == 1) # it has been accepted, it must be served

                tc.getTimeOff(text = '       ')

                # ##### DELAY constraint  #######################################
                
                ## for new jobs
                for jj in range(self.J):
                    for nn in range(self.parent.N):
                        for tt in range(self.parent.T_ext):
                            self.model.addCons(x[jj][nn][tt]*(self.wlessDel_j[jj] +  self.propgDel_max) <= self.parent.MAX_1WAY_DELAY)
                            

                ##  for jobs in system
                for jj in range(self.J, self.J + self.J_in):
                    for nn in range(self.parent.N):
                        for tt in range(self.parent.T_ext):
                            self.model.addCons(x[jj][nn][tt]*(self.wdlay_in[jj - self.J] +  self.propgDel_max) <= self.parent.MAX_1WAY_DELAY)
                            

                ######################################################################
                #%% ---- DEFINING OBJECTIVE function #################################
                ######################################################################
                p.print_('    6 - Joint objective function')

                if self.parent.revcost_function:
                    objFunction = (actual_rev - P_brown_used*self.parent.cost_pw)
                else:
                    if self.parent.nologInFairness:
                        # objFunction = quickprod([OBJ_P,OBJ_R,OBJ_D]) ### it takes too much time
                        objFunction = scipexp(sciplog(OBJ_P) + sciplog(OBJ_R)) #+ sciplog(OBJ_D))
                    else:
                        objFunction = sciplog(OBJ_P) + sciplog(OBJ_R) #+ sciplog(OBJ_D) #+ (-obj_lb) # adding lowerbound to have positive values
                        # objFunction = sciplog(OBJ_P*OBJ_R*OBJ_D) + (-obj_lb) # adding lowerbound to have positive values

                #####################################################################
                #%% ---add objective, params   ------------------------------------#
                #####################################################################
                p.print_('\n Setting objective function, linearized\n')

                if self.parent.revcost_function:
                    lb_no_rev_all_brown = - self.parent.cost_pw * np.amax([0.0, np.sum(self.parent.max_power_node)* self.parent.T_ext - np.sum(self.greenPW)])
                    linearize_var = self.model.addVar(lb=lb_no_rev_all_brown, vtype='C', name='objective_aux_var')
                else:
                    if self.parent.nologInFairness:
                        linearize_var = self.model.addVar(lb=0, ub=1, vtype='C', name='objective_aux_var')
                    else: # added obj_lb to the obj, to have positive values. then ub = -obj_lb
                        linearize_var = self.model.addVar(lb=None, ub=0.01, vtype='C', name='objective_aux_var')

                self.model.setObjective(linearize_var, sense='maximize')
                self.model.addCons(linearize_var <= objFunction)
                self.model.hideOutput(not talkative)

                self.model.setPresolve(pyscipopt.SCIP_PARAMSETTING.AGGRESSIVE) ## Call if you want heuristic more aggresive
                #FAST, DEFAULT, OFF)
                # self.model.writeProblem('Prob_def.cip')

                ############################################################################
                #%% --    SOLVE    ---------------------------------------------------#
                ############################################################################
                p.print_('\n ----  OPTIMIZING -------\n')
                time.sleep(.5)

                print('Previous state:\n', self.parent.state)

                self.model, probStatus = self.parent.optimize(self.model, talkative, iteration = 0, prevGap = float('inf'),
                                             gapDiffStop = self.parent.gapIncrementStop)
                # time.sleep(.5)
                # self.model.writeProblem()

                self.varsDict = {'x': x_np, 'c': c_np, 'y': y_np, 'a': a_j} # 'obj_P': OBJ_P, 'obj_R': OBJ_R, 'obj_D': OBJ_D

                self.sol = self.getSolution()

            return (self.model, self.sol, self.parent.state)

        #%%
        def getSolution(self, doPrintSolution=True):
            """
                Extracts numerical results from SCIP variables, updates the solver state for
                the subsequent sliding window step, and formats return dictionaries.
            """

            probStatus = self.model.getStatus()
            # text_save = Solver.checkStatusSolution(probStatus)
            if probStatus == 'infeasible':
                sys.exit('Probem is infeasible')
            else:
                bestSolution = self.model.getBestSol()
                gap          = self.model.getGap()
                x_sol = Solver.round2int(Solver.getValVars(self.model, self.varsDict['x']))
                y_sol = Solver.round2int(Solver.getValVars(self.model, self.varsDict['y']))
                # y_sol = np.zeros(x_sol.shape)
                a_sol = Solver.round2int(Solver.getValVars(self.model, self.varsDict['a']))

                # c_sol = Solver.getValVars(self.model, self.varsDict['c']) #if not onlyXnotC else c_np
                c_sol = self.given_C # now is not a variable

                objSol  = self.model.getObjVal()

                ############  ############   ############  ############
                ### value of solution --- revenue
                num_arrived    = Counter(self.arriv_t).get(0)
                num_arrived    = num_arrived if num_arrived != None else 0
                num_accepted   = int(np.sum([a_sol[jj] for jj in range(self.J) if self.arriv_t[jj] == 0]))
                num_accepted   = 0 if num_arrived == 0 else num_accepted

                if self.J > 0:
                    accepted_now   = [a_sol[jj] if self.arriv_t[jj] == 0 else 0 for jj in range(self.J)]
                    rev_now_acc    = np.inner(self.rev_j, accepted_now)
                    rev_now_tot    = np.inner(self.rev_j, [1 if self.arriv_t[jj] == 0 else 0 for jj in range(self.J)])
                else:
                    rev_now_tot = 0
                    rev_now_acc = 0

                actual_rev_sol = np.inner(self.revenue_currentJobs, a_sol)
                total_revenue  = np.sum(self.revenue_currentJobs)
                revRatio_sol   = actual_rev_sol/total_revenue if total_revenue != 0 else 1

                OBJ_R_sol      = revRatio_sol * self.parent.rho_r + (1 - self.parent.rho_r)if total_revenue != 0 else 1

                ############  ############   ############  ############
                ############  ############   ############  ############
                #### Updating state of solver
                ############  ############   ############  ############

                newRemainingPCs  = []
                new_C_j_in       = []
                newRemainingTSs  = []
                newCurrentNodes  = []
                new_Rev_j_in     = []
                new_wdlay_in     = []
                delay_tt         = []
                n_vec = np.arange(self.parent.N)

                ############## jobs finished ##############
                for jj in range(self.J + self.J_in):
                    if jj >= self.J or self.arriv_t[jj] == 0: # if arrived now or already in the system
                        # print('arrived now or before')
                        C_jj            = self.C_j[jj]  if jj < self.J else self.C_j_in[jj - self.J]
                        rev_jj_t        = self.rev_j_t_all[jj]
                        wdlay_jj        = self.wlessDel_j[jj] if jj < self.J else self.wdlay_in[jj - self.J]

                        prevCurrentPCs  = [0]*(self.dur_j[jj]-1) + [np.sum(c_sol[jj,:,0]*x_sol[jj,:,0])] if jj < self.J \
                                            else self.previousPCs[jj-self.J][1:] + [np.sum(c_sol[jj,:,0]*x_sol[jj,:,0])]
                        currentTSs_jj   = self.dur_j[jj] if jj < self.J else self.remainingTSs[jj - self.J]
                        currentNode_jj  = np.inner(n_vec,x_sol[jj,:,0]) if x_sol[jj,:,0].sum() > 0 else None
                        # print(currentTSs_jj, jj, self.J)
                        if currentNode_jj !=  None and currentTSs_jj > 1: #10**-3: ## save jobs in jobs in the system
                            new_C_j_in.append(C_jj)
                            newRemainingPCs.append(prevCurrentPCs)
                            newRemainingTSs.append(currentTSs_jj - 1)
                            newCurrentNodes.append(currentNode_jj) # next time, the node is fixed in this iteration
                            new_Rev_j_in.append(rev_jj_t)       # revenue per time for his job
                            new_wdlay_in.append(wdlay_jj)     # wireless delay for his job
                        else:
                            pass

                ############## power ##############
                pw_tot_tt = np.array([[(self.parent.alpha*c_sol[jj,nn,0] + self.parent.gamma)*x_sol[jj,nn,0]
                                        + self.parent.beta*y_sol[jj,nn,0]
                                      for nn in range(self.parent.N)] for jj in range(self.J + self.J_in)])
                pw_nt_br_tt  = np.maximum(np.sum(pw_tot_tt, axis=0) - self.greenPW[:,0], np.zeros(self.parent.N))
                pw_nt_tot_tt = np.sum(pw_tot_tt, axis=0)

                pcAllocated_nn = np.sum(c_sol[:,:,0]*x_sol[:,:,0], axis = 0)

                # ### saving state for next iteration
                ###
                self.parent.state = {'C_j_in':new_C_j_in, 'previousPCs':newRemainingPCs,
                                'remainingTSs':newRemainingTSs, 'currentNodes':newCurrentNodes,
                                'rev_j_t_in': new_Rev_j_in, 'wdlay_in': new_wdlay_in}

                self.parent.pcAllocated   = self.updateHistoryList(self.parent.pcAllocated,   pcAllocated_nn)
                self.parent.delayFinished = self.updateHistoryList(self.parent.delayFinished, delay_tt)
                self.parent.brownPw       = self.updateHistoryList(self.parent.brownPw,       pw_nt_br_tt)
                self.parent.totalPw       = self.updateHistoryList(self.parent.totalPw,       pw_nt_tot_tt)
                self.parent.revArrived    = self.updateHistoryList(self.parent.revArrived,    rev_now_tot)
                self.parent.revAccepted   = self.updateHistoryList(self.parent.revAccepted,   rev_now_acc)
                self.parent.acceptedJobs  = self.updateHistoryList(self.parent.acceptedJobs,  num_accepted)
                self.parent.arrivedJobs   = self.updateHistoryList(self.parent.arrivedJobs,   num_arrived)
                self.parent.migration_y   = self.updateHistoryList(self.parent.migration_y,   y_sol)
                self.parent.acceptanceX   = self.updateHistoryList(self.parent.acceptanceX,   x_sol)

                #########################################

                ### value of solution --- power
                power_sol = np.array([[[(self.parent.alpha*c_sol[jj,nn,tt] + self.parent.gamma)*x_sol[jj,nn,tt]
                                        + self.parent.beta*y_sol[jj,nn,tt]
                                       for tt in range(self.parent.T_ext)] for nn in range(self.parent.N)]
                                      for jj in range(self.J + self.J_in)])
                pw_nt_sol    = np.sum(power_sol, axis=0)
                pw_nt_br_sol = np.maximum(pw_nt_sol - self.greenPW, np.zeros(self.greenPW.shape))

                ## if normalized by total power
                # OBJ_P_sol  = 1 - self.parent.rho_p*np.sum(pw_nt_br_sol)/(self.parent.max_power_node*self.parent.N*self.parent.T_ext)
                ## if normalized by total brown power
                max_brownPW = np.sum(self.parent.max_power_node)*self.parent.T_ext - np.sum(self.greenPW)
                if max_brownPW == 0:
                    OBJ_P_sol = 1
                else:
                    OBJ_P_sol = 1 - self.parent.rho_p * np.sum(pw_nt_br_sol) / max_brownPW


                if self.parent.revcost_function:
                    print('\nRevenue_sol ',actual_rev_sol )
                    print(f'Cost_sol      {np.sum(pw_nt_br_sol)*self.parent.cost_pw:.2f}')
                    # print('a_sol ', a_sol)
                    # print('x_sol ', x_sol)
                    # print('y_sol ', y_sol)
                else:
                    print('OBJ_R_sol ', OBJ_R_sol)
                    print('OBJ_P_sol ', OBJ_P_sol)
                ############  ############   ############  ############

                self.solution = {'arriv_t': self.arriv_t, 'solVal':objSol, 'objR': OBJ_R_sol,
                                 'objP':OBJ_P_sol, 'rev': actual_rev_sol, 'pw_cost':np.sum(pw_nt_br_sol)*self.parent.cost_pw,
                                 'objD':None, 'gap':gap, 'x':x_sol,
                                 'c': c_sol, 'y':y_sol, 'a_j': a_sol, 'sol':bestSolution}

                ############  ############   ############  ############

                ##### Result is self.solution. Below, just printing summary of solution

                ## Accepted and arrived jobs
                arrived_t = np.array([Counter(self.arriv_t).get(tt) for tt in range(self.parent.T_ext)])
                arrived_t[arrived_t==None] = 0
                accepted_t = np.array([int(np.sum([a_sol[jj] for jj in range(self.J) if self.arriv_t[jj] == tt])) for tt in range(self.parent.T_ext)])

                check_feas_v = [accepted_t[ii] for ii in range(self.parent.T) if arrived_t[ii] == 0 or arrived_t[ii] == None]
                if np.sum(check_feas_v) > 0:
                    sys.err('Something wrong, num accepted cannot be higher than num arrived')

                accepted_t[arrived_t == 0] = 0 # accepted = accepted_t.sum()

                if doPrintSolution:
                    str_accept = [f'{accepted_t[tt]}/{arrived_t[tt]}' for tt in range(self.parent.T_ext)]

                    print('\nValues of admission (x):\n',str_accept)
                    print(f"Obj. function  = {self.solution['solVal']:.2f}")
                    print(f'Admission rate =  {np.sum(a_sol[:self.J])/self.J if self.J!=0 else 1:.3f}')
                    print(f'Revenue metric =  {OBJ_R_sol:.3f}')
                    print(f'Power metric   =  {OBJ_P_sol:.3f}')
                    # print(f'Delay metric   =  {OBJ_D_sol:.3f}\n')

            return self.solution

        ##%% Saving date in files
        def saveSolution(self, filepath_variables=None, text_save = ''):
            """Saves raw solution numpy arrays to disk."""
            if not filepath_variables:
                print('-------------')
                print('Error:')
                print('Please provide filepath_variables to save')
                print('-------------')
            else:
                if not os.path.exists(filepath_variables):
                    os.makedirs(filepath_variables)

                np.save(filepath_variables+'_x_'+'_'+'list_'+ text_save +'.npy', self.solution['x'])
                np.save(filepath_variables+'c_'+'_'+'list_'+ text_save +'.npy', self.solution['c'])
                np.save(filepath_variables+'pw_'+'_'+'list_'+ text_save +'.npy', self.greenPW)
                np.save(filepath_variables+'j_'+'_'+'list_'+ text_save +'.npy', self.joblist)

        def saveStatistics(self, model, filepath_stats=os.path.join('.','stats'), printPrompt=False,
                           save = True, printExceptionStats='True'):
            """Exports solver computational statistics to file."""
            if printPrompt:
                model.printStatistics()
            if save:
                try:
                    filename = checkFile_and_rename(filepath_stats)
                    model.writeStatistics(filename)
                except Exception as expectationRaised:
                    if expectationRaised.errno == 9:
                        if os.path.isfile(filename) and printExceptionStats:
                            print("Exception '[Errno 9] Bad file descriptor' raised but file created. Ignoring it...")
                    else:
                        raise Exception(expectationRaised)
                        ### saving values for evaluation

# %%###########  ############   ############  ############
## From here is just for plotting / printing.
##### Accepted nd arrived jobs  ###########################

def plot_realization(mySolver, jobs, greenPW):
        """
        Generates realization plots showing resource capacities, green power levels,
        allocated processing cycles, and job admission counts over time.
        """

        arrived_t = np.array(mySolver.arrivedJobs)
        accepted_t = np.array(mySolver.acceptedJobs)
        max_accepted_theo = [np.sum(mySolver.C_max_n) / np.mean([jb for ts in jobs for jb in ts])] * mySolver.T_tot
        rejected_t = [r if r > 0 else None for r in arrived_t - accepted_t]

        totalPW = np.array(mySolver.totalPw)
        brownPW = np.array(mySolver.brownPw)
        delayJobs = np.array([d for ts in mySolver.delayFinished for d in ts])

        J_ev = np.sum([1 for ts in jobs for j in ts])
        J_ac = len(delayJobs)
        #############################  #############################
        # Figures
        j_axis = np.arange(1, J_ac + 1)
        t_axis = np.arange(0, mySolver.T_tot)
        c_node = np.array(mySolver.pcAllocated)

        C_node_max_T = [capacity_n * np.ones((mySolver.T_tot)) for capacity_n in mySolver.C_max_n]

        #########
        # fig, axs = plt.subplots(1, 1)
        # axs.plot(t_axis, mySolver.revArrived, marker='o', linewidth=0, label='Revenue of Jobs arrived')
        # axs.set_xlabel('Job'), axs.grid(True),  # axs.set_ylim([0,mySolver.D_max])
        # axs.legend(bbox_to_anchor=(0.5, -0.7), loc="lower center", ncol=2)  # mode="expand",
        # fig.tight_layout()

        fig2, axs2 = plt.subplots(2, 1)
        [axs2[0].plot(t_axis, C_node_max_T[nn],  linestyle='solid', linewidth=2, #Solver.markerList[nn],
                      label=f'Max. PW -- N. {nn}') for nn in range(mySolver.N)]
        axs2[0].set_prop_cycle(None)
        [axs2[0].plot(t_axis, greenPW[nn], Solver.markerList[nn], linestyle='None', linewidth=2,
                      label=f'Green PW -- N. {nn}') for nn in range(mySolver.N)]
        axs2[0].set_prop_cycle(None)
        [axs2[0].plot(t_axis, c_node[:, nn], linestyle='dashed', #Solver.markerList[nn],
                      linewidth=1, label=f'Usage - N. {nn}') for nn in range(mySolver.N)]
        [axs2[0].plot(t_axis, totalPW[:, nn], linestyle='dotted', #Solver.markerList[nn],
                      linewidth=1, label=f'PW - N. {nn}') for nn in range(mySolver.N)]
        axs2[0].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=3)  # mode="expand",
        axs2[0].set_xlabel('Time slot'), axs2[0].grid(True)

        axs2[1].plot(t_axis, arrived_t, linestyle='solid', linewidth=2, label='Num. Arrived')
        axs2[1].plot(t_axis, accepted_t, Solver.markerList[0], linestyle='dashed', linewidth=1, label='Num. Accepted')
        axs2[1].plot(t_axis, rejected_t, Solver.markerList[2], color='red', linestyle='None', linewidth=1,
                     label='Num. Rejected')
        # axs2[1].plot(t_axis, max_accepted_theo, linestyle='dotted', linewidth = 1, label='Avg. Theo. Accepted')
        axs2[1].set_xlabel('Time slot'), axs2[1].grid(True)
        axs2[1].legend(loc='best', ncol=3)


        fig2.tight_layout(), plt.show()
