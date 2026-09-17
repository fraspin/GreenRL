
import numpy as np
import random
import copy, sys
from functools import partial
import time

from auxiliarFunctions import append_
from energy_datasets import GreenEnergyGenerationDataset
from carbonedge import CarbonEdgeHandler


################################################################################
""" ALGORITHM DECISION FUNCTIONS """
################################################################################
        
def random_decision(env):
    """
    Random selection of node for Baseline Random.
    It randomly selects one node from a subset of nodes that are reachable
    within the deadline delay constraints.

    Args:
        env: The simulation environment containing current job and node states.

    Returns:
        action (int): The selected node index (1 to N_SERVERS).
    """
   
    node_arrival     = env.job_list_arrivNode[0]
    wireless_latency = env.jobs_list_wdlay[0]
    prop_delay       = env.matrix_propagation_delay[node_arrival, :]
    
    list_delay = np.zeros(env.N_SERVERS)

    # Calculate total round-trip delay for each server
    for n in range(env.N_SERVERS):
        if env.jobs_list_wdlay != []:
            list_delay[n] = (wireless_latency + env.TRASMISSION_DELAY + prop_delay[n])*2 + env.COMPUTATION_DELAY
        else:
            list_delay[n] =    env.COMPUTATION_DELAY + (env.TRASMISSION_DELAY + prop_delay[n])*2

    # Filter servers that meet the strict deadline constraint
    indices = [index for index, value in enumerate(list_delay) if value < env.DEADLINE_DELAY]

    if indices:
        # Select a random valid index and convert to 1-based action space
        action = random.choice(indices) + 1
    else:
        # Fallback if no nodes are reachable: pick a completely random node
        action = int(random.randint(1, env.N_SERVERS))

    return action  #### once returned add this len(jobs) part into the total action vector

def emptier_decision(env):
    """
    Selection of node for Emptier baseline.
    It greedily selects the reachable node that currently has the most available
    computing resources (lowest load).

    Args:
        env: The simulation environment.

    Returns:
        action (int): The selected node index (1 to N_SERVERS).
    """
   
    node_arrival     = env.job_list_arrivNode[0]
    wireless_latency = env.jobs_list_wdlay[0]
    prop_delay       = env.matrix_propagation_delay[node_arrival, :]

    servers_load = np.zeros(env.N_SERVERS)
    list_delay   = np.zeros(env.N_SERVERS)

    # Calculate current load and delay for all servers
    for n in range(env.N_SERVERS):
        # Available capacity = Maximum capacity - Current usage (raw state)
        servers_load[n] = env.MAX_PC_SERVERS[n] - env.next_state_raw[n]

        list_delay[n] = (wireless_latency + env.TRASMISSION_DELAY  + prop_delay[n])*2 + env.COMPUTATION_DELAY

    # Filter for nodes that meet the deadline
    indices = [index for index, value in enumerate(list_delay) if value < env.DEADLINE_DELAY]
    
    if indices:
        valid_indices = [index for index in indices if index < len(servers_load)]
        if valid_indices:
            # Select the node with the maximum available compute resources
            emptiestServer = max(valid_indices, key=lambda index: servers_load[index]) + 1
        else:
            emptiestServer = int(random.randint(1, env.N_SERVERS))
    else:
        # Fallback if no nodes are reachable
        emptiestServer = int(random.randint(1, env.N_SERVERS))


    action = emptiestServer
    return action

def heuristic_decision(env, isMigrationPhase, node_idx, job_idx):
    """
    Node decision for the GreenH heuristic.
    It identifies reachable nodes and then greedily selects the one with
    the highest amount of free green energy. Operates in both allocation and migration phases.

    Args:
        env: The simulation environment.
        isMigrationPhase (int): 1 if deciding a migration, 0 if deciding a new allocation.
        node_idx (int): Current node index (used for migration delays).
        job_idx (int): Current job index (used for migration delays).

    Returns:
        action (int): The selected node index (1 to N_SERVERS).
    """
    
    obs = env.current_state_dict

    obs_decision = obs['nodes_green']
    node         = obs['index']

    # Calculate free green energy based on the quantized state steps
    freer_green_servers = [env.server_num_steps_green[n] - obs_decision[n] for n in range(env.N_SERVERS)]

    # Determine delays depending on whether it is a migration or a new arrival
    if isMigrationPhase == 1:
        Wireless_link_latency = env.DelayList[node_idx][job_idx]
        prop_delay            = env.matrix_propagation_delay[node - 1, :]
    else: ## acceptance
        Wireless_link_latency = env.jobs_list_wdlay[0]
        node_arrival          = env.job_list_arrivNode[0]
        prop_delay            = env.matrix_propagation_delay[node_arrival - 1, :]

    list_delay = np.zeros(env.N_SERVERS)
    
    for n in range(env.N_SERVERS):
        list_delay[n] = (Wireless_link_latency  + prop_delay[n] + env.TRASMISSION_DELAY)*2 + env.COMPUTATION_DELAY

    # Filter for nodes that meet the deadline
    indices = [index for index, value in enumerate(list_delay) if value < env.DEADLINE_DELAY]
    
    if indices:
        valid_indices = [index for index in indices if index < len(freer_green_servers)]
        if valid_indices:
            # Select the node with the maximum available green energy
            action = max(valid_indices, key=lambda index: freer_green_servers[index]) + 1
        else:
            action = int(random.randint(1, env.N_SERVERS))
    else:
        # Fallback if no nodes are reachable
        action = int(random.randint(1, env.N_SERVERS))

    return action

def RL_decision(env, my_model, deterministic_action):
    """
        Wrapper for Stable Baselines3 model predictions.

        Args:
            env: The simulation environment.
            my_model: The trained RL agent (A2C or DQN).
            deterministic_action (bool): Whether to bypass exploration and pick the argmax action.

        Returns:
            action (int): The model's chosen action.
        """

    my_obs          = env.current_state_dict # obs == self.current_state_dict
    action, _states = my_model.predict(my_obs, deterministic=deterministic_action)

    return action


################################################################################
""" MODIFICATIONS CARBONEDGE """
################################################################################

############################################################
from dataclasses import dataclass

@dataclass
class StoredTSData:
    """Dataclass holding the state of jobs and energy for a single timeslot."""
    jobs_pc          : float
    jobs_duration    : int
    jobs_wdlay       : float
    jobs_arrivNd     : int
    greenPW          : any
    
@dataclass
class StoredDataPerTS:
    """Dataclass holding arrays of system state data across all timeslots."""
    jobs_pc          : list
    jobs_duration    : list
    jobs_wdlay       : list
    jobs_arrivNd     : list
    greenPW          : any
    prop_delay       : np.ndarray

    def at(self, t):
        """Returns the specific data snapshot for a given timeslot 't'."""
        return StoredTSData(
            self.jobs_pc[t],
            self.jobs_duration[t],
            self.jobs_wdlay[t],
            self.jobs_arrivNd[t],
            self.greenPW[:, t] if self.greenPW.ndim > 1 else self.greenPW[t],
        )
 

################################################################################
""" OTHER FUNCTIONS TO MODIFY ENVIRONMENT """
################################################################################
        
def updateEnvObservation(env):
    """
    Updates the environment's observation state safely.
    Used to progress the state when no jobs arrive and no actions are taken.
    """
    env.next_state_quantized_dict = env.state_quantization(env.next_state_raw)
    env.current_state_dict        = env.next_state_quantized_dict
    
    return env.current_state_dict


################################################################################
""" MAIN FUNCTION TO EVALUATE ANY ALGORITHM """
################################################################################
                    
def evaluate_algorithm(algorithm, env, data_experiment, results_policy, model=None):
    """
    Main evaluation loop handling the execution of the selected algorithm over
    the defined number of realizations (experiments) and timeslots.

    Args:
        algorithm (str): Name of the algorithm to evaluate ('random', 'emptier', 'sota', 'policy', etc.).
        env: The simulation environment.
        data_experiment (dict): Configuration dictionary loaded from YAML.
        results_policy (list): Saved policy execution results to ensure exact workload reproduction for baselines.
        model: Optional trained RL model if evaluating 'sota' or 'policy'.

    Returns:
        results_eval (list): A list of dictionaries containing metrics for each realization.
    """
    # Retrieve overall evaluation parameters
    tot_time_evaluation = data_experiment['tot_time_evaluation']
    n_realizations      = data_experiment['n_real_eval']
    window_size         = data_experiment['evaluating_window_size']
    TRANSIENT           = int(tot_time_evaluation * \
                                data_experiment.get('TRANSIENT_PERCT_EVAL', 0)
                                )
    isDeterministic = data_experiment.get('ACTION_DETERMINISTIC', 1)
    NOLOG_FAIRNESS  = data_experiment.get('NOLOG_FAIRNESS', 0)

    policy      = False
    noMigration = True

    # Setup algorithm specific configurations and partial functions
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    if algorithm == 'random':
        print('-----------BASELINE RANDOM ----------------------------')
        decisionFunction = random_decision
    elif algorithm == 'emptier':
        print('-----------BASELINE EMPTIER -----------------------------')
        decisionFunction = emptier_decision
    elif algorithm == 'sota':
        print('-----------BASELINE SOTA ----------------------------------')
        decisionFunction = partial(RL_decision, my_model = model, deterministic_action=isDeterministic)
    elif algorithm == 'policy':
        print('-----------POLICY GREENRL ---------------------------------')
        noMigration = False
        policy      = True
        decisionFunction = partial(RL_decision, my_model = model, deterministic_action=isDeterministic)
    elif algorithm == 'heuristic':
        print('-----------HEURISTIC EVALUATION ---------------------------')
        noMigration = False
        decisionFunction = heuristic_decision
    elif algorithm == 'carbonedge':
        print('-----------CARBONEDGE EVALUATION ---------------------------')
        decisionFunction = CarbonEdgeHandler.carbonedge_retrieveDecision
    else:
        sys.err("Unknown algorithm: ', '\nPlease write 'random', 'emptier', 'sota', 'policy' or 'heuristic'")
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------\n')
    

    results_eval = []
    # Iterate over multiple stochastic realizations (episodes)
    for n_time in range(n_realizations):

        if not policy:
            # For baselines, we MUST reproduce the exact same environment (jobs, energy)
            # that the RL policy experienced to ensure fair benchmarking.
            reproduce_scenario = True

            dataTS = StoredDataPerTS(
                    copy.deepcopy(results_policy[n_time]['job_pc_list']),
                    copy.deepcopy(results_policy[n_time]['job_dur_list']),
                    copy.deepcopy(results_policy[n_time]['job_wdlay_list']),
                    copy.deepcopy(results_policy[n_time]['job_arrivNd_list']),
                    copy.deepcopy(results_policy[n_time]['pw_nt_gr_sol']),
                    copy.deepcopy(results_policy[n_time]['propagation_delay'])
                )

            #### Generating the same jobs_pc and energy as the policy_evaluation case
            obs = env.reset(23, reproduce_scenario, dataTS, envNoMigration=noMigration, isTraining=False)

        else:
            # For the RL Policy, we let it generate the environment randomly
            # (which is then saved and used by baselines).
            obs = env.reset(isTraining=False)

        # ---------------------------------------------------------
        # INITIALIZE TRACKING VARIABLES FOR THIS REALIZATION
        # ---------------------------------------------------------

        ### ENERGY VARIABLES
        pw_nt_sol    = np.zeros((env.N_SERVERS, tot_time_evaluation))
        pw_nt_br_sol = np.zeros((env.N_SERVERS, tot_time_evaluation))
        pw_nt_gr_sol = np.zeros((env.N_SERVERS, tot_time_evaluation))
        P_nt         = np.zeros((tot_time_evaluation, env.N_SERVERS))

        ### JOBS RELATED VARIABLES
        rejected_jobs_t                   = np.zeros(tot_time_evaluation)
        accepted_jobs_t                   = np.zeros(tot_time_evaluation)
        num_newArrivedjobs                = np.zeros(tot_time_evaluation)
        rejection_own                     = np.zeros(tot_time_evaluation)
        n_migrations_t                    = np.zeros(tot_time_evaluation)

        rejected_jobs_power_t             = np.zeros(tot_time_evaluation)
        rejected_jobs_deadline_t          = np.zeros(tot_time_evaluation)
        interrupted_jobs_power_t          = np.zeros(tot_time_evaluation)
        interrupted_jobs_delay_t          = np.zeros(tot_time_evaluation)
        interrupted_jobs_wrong_decision_t = np.zeros(tot_time_evaluation)
        
        capacity_violation_rejected_new_jobs_t     = np.zeros(tot_time_evaluation)
        deadline_violation_rejected_new_jobs_t     = np.zeros(tot_time_evaluation)
        mig_delay_violation_rejected_new_jobs_t    = np.zeros(tot_time_evaluation)
        capacity_violation_mig_interrupted_jobs_t  = np.zeros(tot_time_evaluation)
        mig_delay_violation_mig_interrupted_jobs_t = np.zeros(tot_time_evaluation)
        migration_violation_interrupted_jobs_t     = np.zeros(tot_time_evaluation)

        total_revenue_j    = [[]] * tot_time_evaluation
        revenue_j          = [[]] * tot_time_evaluation
        penalty            = [[]] * tot_time_evaluation

        job_pc_list        = [[]] * tot_time_evaluation
        job_ts_list        = [[]] * tot_time_evaluation
        job_wd_list        = [[]] * tot_time_evaluation ## wireless delay of new job
        job_an_list        = [[]] * tot_time_evaluation ## arrival node of new job


        ## OTHER VARIABLES/INDEXES
        total_jobs    = 0
        carbonedge_emissions_t = np.zeros(tot_time_evaluation) ## for carbonedge - emissions tracking

        # ---------------------------------------------------------
        # START TIMESLOT ITERATION
        # ---------------------------------------------------------
        for t_slot in range(tot_time_evaluation):
            # Initialize variables for migration 
            mig_power_violation   = 0
            mig_delay_violation   = 0
            mig_reject_violation  = 0
            flag_migrationPhase   = 0
            n_of_migration        = np.zeros(env.N_SERVERS)

            rejected_num_jobs     = 0
            revenueAcceptedJobs_t = []
            revenueTotalJobs_t    = []
            penalty_t             = []
            
            # --- Extract/Log New Arrival Data ---
            if policy:
                numNewJobs       = len(env.jobs_list_timeS)
                isAccepted       = np.zeros(numNewJobs)
                jobList_aux      = copy.deepcopy(env.jobs_list_timeS)

                # Save generated job states to reuse for baselines later
                job_pc_list[t_slot]   = [env.c_jn for job in env.jobs_list_timeS]
                job_ts_list[t_slot]   = [job for job in env.jobs_list_timeS]
                job_wd_list[t_slot]   = [job for job in env.jobs_list_wdlay]
                job_an_list[t_slot]   = [node for node in env.job_list_arrivNode]

                # Update Green Energy every 45 timeslots (simulating 15 real-world minutes)
                if t_slot % 45 == 0 and t_slot != 0: # 45 timeslots means 15 minutes in real life
                    env.next_state_raw[env.N_SERVERS: 2 * env.N_SERVERS] = GreenEnergyGenerationDataset(env.N_SERVERS, env.MAX_PW_SERVERS, round(t_slot/45))
                    env.next_state_quantized_dict = env.state_quantization(env.next_state_raw)
                    env.current_state_dict = env.next_state_quantized_dict

            else:
                # Load the pre-recorded jobs for the baseline run
                print(f"DEBUG: t_slot value is {t_slot}")
                print(f"DEBUG: length of dataTS.jobs_pc is {len(dataTS.jobs_pc)}")
                print('dataTS.jobs_pc[t_slot]',dataTS.jobs_pc[t_slot])
                numNewJobs       = len(dataTS.jobs_pc[t_slot])
                isAccepted       = np.zeros(numNewJobs)
                jobList_aux      = copy.deepcopy(dataTS.jobs_duration[t_slot])

                if t_slot % 45 == 0 and t_slot != 0: # 45 timeslots means 15 minutes in real life
                    env.next_state_raw[env.N_SERVERS: 2 * env.N_SERVERS] = dataTS.greenPW[:, t_slot]
                    env.next_state_quantized_dict = env.state_quantization(env.next_state_raw)
                    env.current_state_dict = env.next_state_quantized_dict

            startNodeList_vectorized = np.hstack(env.startTS_Node_list)

            # Check environment context to determine if we are in migration or allocation
            if t_slot == 0:
                obs_dict = obs[0]
                index_migration = obs_dict["index"]
            else:
                index_migration = obs.get('index')
            if noMigration: ## this algorithm do not do migrations
                nothingToDoThisTS = numNewJobs == 0 # no new jobs = nothing to do
                doMigrationPhase  = False ## there is no migration phase
                
            else: ## for algorithms which do migrations
                ## if no new jobs and no jobs in the system = nothing to do
                nothingToDoThisTS = numNewJobs == 0 and index_migration == 0
                if t_slot ==0 and numNewJobs == 0:
                    nothingToDoThisTS = 1 #to avoid bug for policy when no jobs are extracted at time = 0

                ## if there are nodes in the system, we do migration
                doMigrationPhase  = index_migration != 0 
                
            if nothingToDoThisTS: ## nothing to do in this time slot: move next.
                env.handleChangeTS()
                obs = updateEnvObservation(env)

            else:
                # ==========================================
                # MIGRATION PHASE
                # ==========================================
                if noMigration: 
                    if index_migration != 0: # move state to accept new jobs
                        env.current_state_dict['index'] = 0
                        env.next_state_raw[2 * env.N_SERVERS] = 0   
                        
                if doMigrationPhase:
                    flag_migrationPhase = 1
                    for node_idx, nodeJobs in enumerate(env.startTS_Node_list):
                        if nodeJobs != []:
                            for job_idx, jobLength in enumerate(nodeJobs):

                                ####### take decision ###################                                
                                if algorithm == 'heuristic':
                                    action = decisionFunction(env, flag_migrationPhase, node_idx, job_idx)
                                else:
                                    action = decisionFunction(env)
                                    
                                ####### apply decision ##################
                                obs, rewards, done, truncated, info = env.step(action)

                                ### check outcome ########################
                                if env.lastJobWasInterrupted == 1:
                                    print('INTERRUPTED --ERROR IN MIGRATION VALUE')
                                    mig_reject_violation += 1
                                    # mig_reject_violation = env.error_migration
                                    penalty_t.append(env.REV_FACTOR * env.c_jn * env.jobSessionLength)


                                if env.lastJobExceededNodePw == 1:  # violation exceeding the node power
                                    print('INTERRUPTED -- ERROR IN EXCEEDING NODE POWER DURING MIGRATION')
                                    mig_power_violation += 1
                                    penalty_t.append(env.REV_FACTOR * env.c_jn * env.jobSessionLength)

                                if env.lastJobExceededDeadline == 1:  # violation exceeding deadline
                                    print('INTERRUPTED -- ERROR IN EXCEEDING DEADLINE DURING MIGRATION')
                                    mig_delay_violation += 1
                                    penalty_t.append(env.REV_FACTOR * env.c_jn * env.jobSessionLength)

                        else: 
                            # HERE WE DO NOTHING OTHERWISE IT BREAKS THE FLOW
                            pass

                    n_of_migration = [int(env.migration[node_idx]) for node_idx in range(env.N_SERVERS)]

                # ==========================================
                # ALLOCATION PHASE
                # ==========================================
                if numNewJobs != 0:
                    flag_migrationPhase = 0

                    if algorithm == 'carbonedge':                        
                        carbonedge_algorithm = CarbonEdgeHandler(
                            env, 
                            data_experiment,
                            dataTS.at(t_slot)
                        )
                        
                        carbonedge_algorithm.optimize()

                        carbonedge_emissions_t[t_slot] = carbonedge_algorithm.get_carbon_emissions()

                    for j in range(numNewJobs):
                        ####### take decision ###################
                        if algorithm == 'heuristic':
                            action = decisionFunction(env, flag_migrationPhase, None, None)
                        elif algorithm == 'carbonedge':
                            action = decisionFunction(carbonedge_algorithm, j)
                        else:
                            action = decisionFunction(env)       
        
                        ####### apply decision ##################
                        obs, rewards, done, truncated, info = env.step(action)
    
                        ### check outcome ########################
                        if action != 0:
                            if env.lastJobExceededNodePw == 0 and env.lastJobExceededDeadline == 0:  
                            # job has been really accepted
                                isAccepted[j] = 1
                                revenueAcceptedJobs_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                            else:  # job rejected due to capacity violation
                                if env.lastJobExceededNodePw == 1: # violation exceeding the node power
                                    print(f'REJECTED new job {j} at t={t_slot} EXCEEDING NODE POWER')
                                    penalty_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                                    rejected_jobs_power_t[t_slot] += 1
                                elif env.lastJobExceededDeadline == 1:
                                    print(f'REJECTED new job {j} at t={t_slot} EXCEEDING DEADLINE')
                                    penalty_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                                    rejected_jobs_deadline_t[t_slot] += 1
                                else:
                                    print(f'++++\nWARNING!!!!!! new job {j} at t={t_slot} SHOULD NOT BE HERE')    
                        else:
                            rejected_num_jobs += 1
    
                        total_jobs += 1
                        revenueTotalJobs_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                        num_newArrivedjobs[t_slot] += 1
                else:
                    pass ### corresponding changes happen in environ, inside last env.step

            # ---------------------------------------------------------
            # END OF TIMESLOT METRIC AGGREGATION
            # ---------------------------------------------------------
            nodesPW = env.getPowerConsumption()

            revenue_j[t_slot]       = revenueAcceptedJobs_t
            total_revenue_j[t_slot] = revenueTotalJobs_t
            penalty[t_slot]         = penalty_t

            if np.sum(startNodeList_vectorized) == 0:
                n_migrations_t[t_slot] = 0
            else:
                n_migrations_t[t_slot] = np.sum(n_of_migration)

            accepted_jobs_t[t_slot] = np.sum(isAccepted)
            rejection_own[t_slot]   = rejected_num_jobs  ## rejected because of fair decision
            rejected_jobs_t[t_slot] = rejected_num_jobs + rejected_jobs_power_t[t_slot] + rejected_jobs_deadline_t[t_slot]

            interrupted_jobs_wrong_decision_t[t_slot] = mig_reject_violation ##  includes delay violations so far
            interrupted_jobs_power_t[t_slot]          = mig_power_violation
            interrupted_jobs_delay_t[t_slot]          = mig_delay_violation

            capacity_violation_rejected_new_jobs_t[t_slot]     = rejected_jobs_power_t[t_slot]
            deadline_violation_rejected_new_jobs_t[t_slot]     = rejected_jobs_deadline_t[t_slot]

            # the following ones are duplicated and should be removed. 
            mig_delay_violation_rejected_new_jobs_t[t_slot]    = deadline_violation_rejected_new_jobs_t[t_slot]

            migration_violation_interrupted_jobs_t[t_slot]     = interrupted_jobs_wrong_decision_t[t_slot]
            capacity_violation_mig_interrupted_jobs_t[t_slot]  = interrupted_jobs_power_t[t_slot]
            mig_delay_violation_mig_interrupted_jobs_t[t_slot] = interrupted_jobs_delay_t[t_slot]
 
            ## saving energy consumption
            for nn in range(env.N_SERVERS):
                #####---EQ: 5---####
                brown_power = max(nodesPW[nn] - env.next_state_raw[env.N_SERVERS + nn], 0)
                P_nt[t_slot, nn]         = brown_power
                pw_nt_sol[nn, t_slot]    = nodesPW[nn]
                pw_nt_br_sol[nn, t_slot] = brown_power
                pw_nt_gr_sol[nn, t_slot] = env.next_state_raw[env.N_SERVERS + nn]

        # ---------------------------------------------------------
        # FINAL METRICS COMPILATION FOR REALIZATION
        # ---------------------------------------------------------
        results_eval_n = {}
        results_eval_n['pw_nt_sol']    = pw_nt_sol
        results_eval_n['pw_nt_br_sol'] = pw_nt_br_sol
        results_eval_n['pw_nt_gr_sol'] = pw_nt_gr_sol
        results_eval_n['cost_br_pw']   = np.sum(pw_nt_br_sol,axis =0)*env.COST_FACTOR

        results_eval_n['total_brown_used']      = np.sum(pw_nt_br_sol)
        results_eval_n['total_green_available'] = np.sum(pw_nt_gr_sol)
        results_eval_n['total_green_not_used']  = np.sum(pw_nt_gr_sol) - (np.sum(pw_nt_sol) - np.sum(pw_nt_br_sol))
        
        results_eval_n['accepted_jobs_t']       = accepted_jobs_t
        results_eval_n['rejected_jobs_t']       = rejected_jobs_t ## sum of the following 3
        results_eval_n['rejected_jobs_power_t'] = rejected_jobs_power_t
        results_eval_n['rejected_jobs_delay_t'] = rejected_jobs_deadline_t
        results_eval_n['rejection_own']         = rejection_own # jobs for which algorithm decides not to  accept them -- not due to errors 

        results_eval_n['interrupted_rejected_t'] = interrupted_jobs_wrong_decision_t # rejections when migrating -- included because of deadline
        results_eval_n['interruptedpower_t']     = interrupted_jobs_power_t # rejecting due to migration to full node
        results_eval_n['interruptedelay_t']      = interrupted_jobs_delay_t # rejecting due to migration to full node

        results_eval_n['n_migrations_t']        = n_migrations_t
        results_eval_n['n_migrations_t_total']  = np.mean(n_migrations_t)

        if policy:
            results_eval_n['job_pc_list']       = job_pc_list
            results_eval_n['job_dur_list']      = job_ts_list
            results_eval_n['job_wdlay_list']    = job_wd_list
            results_eval_n['job_arrivNd_list']  = job_an_list
            results_eval_n['propagation_delay'] = env.matrix_propagation_delay
        else:
            results_eval_n['job_pc_list']       = dataTS.jobs_pc
            results_eval_n['job_dur_list']      = dataTS.jobs_duration
            results_eval_n['job_wdlay_list']    = dataTS.jobs_wdlay
            results_eval_n['job_arrivNd_list']  = dataTS.jobs_arrivNd
            results_eval_n['propagation_delay'] = env.matrix_propagation_delay

        results_eval_n['migration_violation_interrupted_jobs']     = np.sum(migration_violation_interrupted_jobs_t)
        results_eval_n['capacity_violation_rejected_new_jobs']     = np.sum(capacity_violation_rejected_new_jobs_t)
        results_eval_n['mig_delay_violation_rejected_new_jobs']    = np.sum(mig_delay_violation_rejected_new_jobs_t)
        results_eval_n['capacity_violation_mig_interrupted_jobs']  = np.sum(capacity_violation_mig_interrupted_jobs_t)
        results_eval_n['mig_delay_violation_mig_interrupted_jobs'] = np.sum(mig_delay_violation_mig_interrupted_jobs_t)
        results_eval_n['deadline_violation_rejected_new_jobs']     = np.sum(deadline_violation_rejected_new_jobs_t)

        # CarbonEdge specific metrics
        results_eval_n['carbonedge_emissions_t']     = carbonedge_emissions_t
        results_eval_n['carbonedge_total_emissions'] = np.sum(carbonedge_emissions_t)

        results_eval_n = getResults(results_eval_n, env, P_nt, pw_nt_gr_sol, tot_time_evaluation, window_size,
                                    num_newArrivedjobs, accepted_jobs_t, total_jobs,  revenue_j, penalty,
                                    total_revenue_j, TRANSIENT, NOLOG_FAIRNESS)

        results_eval = append_(results_eval, results_eval_n)

    return results_eval


def getResults(results_eval_nn, env, P_nt, pw_nt_gr_sol, tot_time_evaluation,
                window_size, num_newArrivedjobs, accepted_jobs_t, total_jobs, revenue_j, penalty, total_revenue_j, TRANSIENT, NOLOG_FAIRNESS):
        """
        Calculates Key Performance Indicators (KPIs) over sliding windows and over
        the total duration of the experiment, adjusting for the warm-up (transient) period.
        """
        ###### POWER Metric ##############################
        acceptanceratio_moving_average    = []
        revenueratio_moving_average       = []
        revenue_moving_average            = []
        pwTerm_fairness_movAv             = []
        revTerm_fairness_movAv            = []
        obj_func_fairness_moving_average  = []
        delay_moving_average              = []
        revenue_cost_moving_average       = []
        theo_bound_revenue_moving_average = []
        obj_func_revenue_cost_moving_average = []

        Power_brown_per_time = np.sum(P_nt, axis=1)
        Power_green_per_time = np.sum(pw_nt_gr_sol, axis=0)
        cost_brown_per_time  = env.COST_FACTOR * Power_brown_per_time

        P_total_ma = (np.sum(env.MAX_PW_SERVERS) - np.sum(env.green_power_levels)) * window_size

        cc_jn = env.alpha * env.c_jn + env.gamma
        average_total_used_power_time_slot = [cc_jn * env.arrivalRate] * tot_time_evaluation

        # ---------------------------------------------------------
        # SLIDING WINDOW METRICS
        # Calculate moving averages over specified window_size
        # ---------------------------------------------------------
        i = 0
        while i < tot_time_evaluation - window_size + 1:
            # Calculate the average of current window
            if np.sum(num_newArrivedjobs[i:i + window_size]) != 0:
                AcceptanceRatio_t1     = round(np.sum(accepted_jobs_t[i:i + window_size]) / (np.sum(num_newArrivedjobs[i:i + window_size])), 2)
                revenue_i_window       = [np.sum(dd) for dd in revenue_j[i:i + window_size]]
                if penalty == None:
                    penalty_i_window = [0]
                    penalty = [0]
                else:
                    penalty_i_window       = [np.sum(dd) for dd in penalty[i:i + window_size]]
                total_revenue_i_window = [np.sum(dd) for dd in total_revenue_j[i:i + window_size]]

                revenue_moving_average_eval = np.sum(revenue_i_window)  -  np.sum(penalty_i_window)
                revRat_moving_average_eval  = revenue_moving_average_eval/ np.sum(total_revenue_i_window)

                pw_brown_window   = np.sum(Power_brown_per_time[i:i + window_size])
                cost_brown_window = np.sum(cost_brown_per_time[i:i + window_size])

                if P_total_ma == 0:
                    pwTerm_fairness_movAv_eval = 1
                else:
                    pwTerm_fairness_movAv_eval  = 1 - env.rho_p * (pw_brown_window / P_total_ma)

                revTerm_fairness_movAv_eval = revRat_moving_average_eval * env.rho_r + (1 - env.rho_r)

                revenue_minus_cost_average_eval = revenue_moving_average_eval - cost_brown_window
                obj_function_revenue_cost_eval  = revenue_minus_cost_average_eval/np.sum(total_revenue_i_window)

                #### for the theoretic bound...
                total_cost_brown_window = max(np.sum(average_total_used_power_time_slot[i:i + window_size] - Power_green_per_time[i:i + window_size]),0)
                theo_bound_revenue_eval = np.sum(total_revenue_i_window) - env.COST_FACTOR * total_cost_brown_window

            else:
                AcceptanceRatio_t1              = 0
                pwTerm_fairness_movAv_eval      = 0
                revTerm_fairness_movAv_eval     = 0
                revRat_moving_average_eval      = 0
                revenue_moving_average_eval     = 0
                revenue_minus_cost_average_eval = 0
                theo_bound_revenue_eval         = 0
                obj_function_revenue_cost_eval  = 0

            obj_func_revenue_cost_moving_average.append(obj_function_revenue_cost_eval)
            acceptanceratio_moving_average.append(AcceptanceRatio_t1)
            pwTerm_fairness_movAv.append(pwTerm_fairness_movAv_eval)
            revTerm_fairness_movAv.append(revTerm_fairness_movAv_eval)

            revenueratio_moving_average.append(revRat_moving_average_eval)
            revenue_moving_average.append(revenue_moving_average_eval)
            revenue_cost_moving_average.append(revenue_minus_cost_average_eval)
            theo_bound_revenue_moving_average.append(theo_bound_revenue_eval)
            objective_function_ma =  revTerm_fairness_movAv_eval * pwTerm_fairness_movAv_eval

            if NOLOG_FAIRNESS:
                obj_func_fairness_moving_average.append(objective_function_ma)
            else:
                obj_func_fairness_moving_average.append(np.log(objective_function_ma))

            i += 1
            

        # ---------------------------------------------------------
        # TOTAL OVERALL METRICS (Discarding the Transient warmup)
        # ---------------------------------------------------------

        AcceptanceRatioTotal = (np.sum(accepted_jobs_t)) / total_jobs

        ### TRANSIENT PHASE
        revenue_j_list = [item for sublist in revenue_j for item in sublist]
        revenue_j_list = revenue_j_list[TRANSIENT:]

        if penalty != [0]:
            penalty_list = [item for sublist in penalty for item in sublist]
            penalty_list = penalty_list[TRANSIENT:]
            penaltyTotal = np.sum(penalty_list)
        else:
            penaltyTotal = 0

        total_revenue_j_list = [item for sublist in total_revenue_j for item in sublist]
        total_revenue_j_list = total_revenue_j_list[TRANSIENT:]

        if np.sum([np.sum(d) for d in total_revenue_j]) != 0:
           if penalty != [0]:
               RevenueRatioTotal = (np.sum(revenue_j_list) - np.sum(penalty_list) ) / np.sum(total_revenue_j_list)
           else:
               RevenueRatioTotal = (np.sum(revenue_j_list)) / np.sum(total_revenue_j_list)
        else:
            RevenueRatioTotal = 1

        if penalty != [0]:
            OBJ_revenue_cost = (np.sum(revenue_j_list) - np.sum(cost_brown_per_time[TRANSIENT:]) - np.sum(penalty_list)) \
                           / np.sum(total_revenue_j_list)
        else:
            OBJ_revenue_cost = (np.sum(revenue_j_list) - np.sum(cost_brown_per_time[TRANSIENT:]) ) \
                           / np.sum(total_revenue_j_list)

        revenueTotal         = np.sum(revenue_j_list)
        costPWtotal          = np.sum(cost_brown_per_time[TRANSIENT:])

        P = np.sum(P_nt[TRANSIENT:])

        if (np.sum(env.MAX_PW_SERVERS)  - np.sum(env.green_power_levels))  == 0:
            normPower_fair = 1
        else:
            normPower_fair = 1 - env.rho_p * (P / ((np.sum(env.MAX_PW_SERVERS)  - np.sum(env.green_power_levels)) * (tot_time_evaluation - TRANSIENT)  ))

        normRev_fair   = RevenueRatioTotal * env.rho_r + (1 - env.rho_r)

        if NOLOG_FAIRNESS:
            obj_fun = normPower_fair * normRev_fair
        else:
            obj_fun = np.log(normPower_fair * normRev_fair)

        # ---------------------------------------------------------
        # SAVE INTO RESULTS DICTIONARY
        # ---------------------------------------------------------

        results_eval_nn['acceptance_movingAv']         = acceptanceratio_moving_average
        results_eval_nn['revRat_movingAv']             = revenueratio_moving_average
        results_eval_nn['powerFairness_movingAV']      = pwTerm_fairness_movAv
        results_eval_nn['revFairness_movingAV']        = revTerm_fairness_movAv
        results_eval_nn['objFairness_movingAv']        = obj_func_fairness_moving_average
        results_eval_nn['delay_moving_average']        = delay_moving_average

        results_eval_nn['revenue_movingAV']            = revenue_moving_average
        results_eval_nn['revenue_cost_movingAV']       = revenue_cost_moving_average
        results_eval_nn['revenue_cost_movingAV_theo']  = theo_bound_revenue_moving_average
        results_eval_nn['obj_function_revenue_costMV'] = obj_func_revenue_cost_moving_average

        results_eval_nn['objFunction_Fairness'] = obj_fun
        results_eval_nn['normPower_fair']       = normPower_fair
        results_eval_nn['normRev_fair']         = normRev_fair

        results_eval_nn['RevenueRatioTotal'] = RevenueRatioTotal
        results_eval_nn['Objective_function_rev_minus_cost'] = OBJ_revenue_cost
        results_eval_nn['revenueTotal'] = revenueTotal
        results_eval_nn['costPWTotal']  = costPWtotal
        results_eval_nn['penaltyTotal'] = penaltyTotal

        return  results_eval_nn
