import time

import numpy as np
import copy

from auxiliarFunctions import append_
from evaluation_algorithms import  updateEnvObservation, getResults


def evaluate_sota(env, data_experiment, results_policy, model):
    """
        Evaluates the State-of-the-Art (SOTA / CC23) algorithm over multiple scenario realizations.

        To ensure a fair benchmark comparison, this function reproduces the exact same workload
        (job arrivals, compute requirements, wireless delays, node locations, and green energy availability)
        that was generated during the GreenRL policy evaluation run.

        Args:
            env (WorldEnvStateOfTheArt): The SOTA Gymnasium simulation environment instance.
            data_experiment (dict): Configuration dictionary loaded from the experiment YAML file.
            results_policy (list): List of dictionaries containing the workload and energy traces
                                   recorded during the GreenRL policy evaluation.
            model (StableBaselines3 Model): The trained SOTA RL model (e.g., DQN).

        Returns:
            results_eval (list): A list of dictionaries containing evaluation metrics for each realization.
    """

    print('---------------------------------------')
    print('---------------------------------------')
    print('---------------------------------------')
    print('-----------STATE-OF-THE-ART------------')
    print('---------------------------------------\n')
    print('---------------------------------------\n')
    print('---------------------------------------\n')

    # Extract general evaluation hyperparameters from the experiment config
    tot_time_evaluation = data_experiment['tot_time_evaluation']
    n_realizations      = data_experiment['n_real_eval']
    window_size         = data_experiment['evaluating_window_size']
    TRANSIENT           = int(tot_time_evaluation * \
                                data_experiment.get('TRANSIENT_PERCT_EVAL', 0)
                                )
    isDeterministic = data_experiment.get('ACTION_DETERMINISTIC', 1)
    NOLOG_FAIRNESS  = data_experiment.get('NOLOG_FAIRNESS', 0)

    results_eval = []

    # Iterate over each independent evaluation realization (simulation run)
    for n_time in range(n_realizations):

        # Retrieve exact scenario data from the GreenRL policy run for fair benchmarking
        jobs_pc       = copy.deepcopy(results_policy[n_time]['job_pc_list'])
        jobs_duration = copy.deepcopy(results_policy[n_time]['job_dur_list'])
        jobs_wdlay    = copy.deepcopy(results_policy[n_time]['job_wdlay_list'])
        jobs_arrivNd  = copy.deepcopy(results_policy[n_time]['job_arrivNd_list'])
        prop_delay    = copy.deepcopy(results_policy[n_time]['propagation_delay'])
        greenPW       = copy.deepcopy(results_policy[n_time]['pw_nt_gr_sol'])
        
        # Reset the environment using fixed seeds and deterministic pre-generated data traces
        obs = env.reset(23, 1, jobs_duration, greenPW, prop_delay, jobs_wdlay,jobs_arrivNd, isTraining=False)
        obs = obs[0]

        # ---------------------------------------------------------
        # INITIALIZE ENERGY & JOB METRICS TRACKERS FOR THIS REALIZATION
        # ---------------------------------------------------------
        pw_nt_sol      = np.zeros((env.N_SERVERS, tot_time_evaluation))
        pw_nt_br_sol   = np.zeros((env.N_SERVERS, tot_time_evaluation))
        pw_nt_gr_sol   = np.zeros((env.N_SERVERS, tot_time_evaluation))
        P_nt           = np.zeros((tot_time_evaluation, env.N_SERVERS))

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

        capacity_violation_rejected_new_jobs_t    = np.zeros(tot_time_evaluation)
        deadline_violation_rejected_new_jobs_t    = np.zeros(tot_time_evaluation)
        capacity_violation_mig_interrupted_jobs_t = np.zeros(tot_time_evaluation)
        deadline_violation_mig_interrupted_jobs_t = np.zeros(tot_time_evaluation)
        migration_violation_interrupted_jobs_t    = np.zeros(tot_time_evaluation)

        total_revenue_j    = [[]] * tot_time_evaluation
        revenue_j          = [[]] * tot_time_evaluation
        job_pc_list        = [[]] * tot_time_evaluation
        job_ts_list        = [[]] * tot_time_evaluation
        job_wd_list        = [[]] * tot_time_evaluation ## wireless delay of new job
        job_an_list        = [[]] * tot_time_evaluation ## arrival node of new job
        penalty            = [[]] * tot_time_evaluation
        
        total_jobs    = 0

        # ---------------------------------------------------------
        # MAIN EVALUATION LOOP (TIMESLOT BY TIMESLOT)
        # ---------------------------------------------------------
        for t_slot in range(tot_time_evaluation):           
            
            mig_power_violation   = 0
            mig_delay_violation   = 0
            mig_reject_violation  = 0
            rejected_num_jobs     = 0
            flag_migrationPhase   = 0
            
            n_of_migration        = 0
            revenueAcceptedJobs_t = []
            revenueTotalJobs_t    = []
            penalty_t             = []
            


            numNewJobs       = len(jobs_pc[t_slot])
            isAccepted       = np.zeros(numNewJobs)
            jobList_aux      = copy.deepcopy(jobs_duration[t_slot])


            nothingToDoThisTS = numNewJobs == 0 # no new jobs = nothing to do

            if nothingToDoThisTS:
                # If no jobs arrive in this time slot, skip processing
                pass
            else:
                # Process each incoming job arrival sequentially
                for j in range(numNewJobs):

                    # 1. Obtain action prediction from the trained SOTA RL model
                    action, _states = model.predict(obs, deterministic=isDeterministic)

                    # 2. Step the SOTA environment with the selected action
                    obs, rewards, done, truncated, info = env.step(action)


                    # 3. Process action outcomes
                    if action == 0:
                        # Action 0: Model explicitly decided to reject the job
                       rejected_num_jobs += 1
                    else:
                        # Model decided to allocate/offload the job to a server node (Action > 0)
                        if  env.lastJobExceededQueue == 0 and env.lastJobExceededDeadline == 0:
                            # Successful allocation (No queue overflow or deadline violation)
                           isAccepted[j] = 1
                           revenueAcceptedJobs_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                           n_migrations_t[t_slot] += env.migration_sota

                        else:
                            # Allocation failed due to constraint violations (illegal model move)
                            if env.lastJobExceededNodePw == 1:
                                # Capacity / Processing Power Exceeded
                               print('job EXCEEDING POWER')
                               penalty_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                               rejected_jobs_power_t[t_slot] += 1

                            elif env.lastJobExceededDeadline == 1:
                                # Execution / Transmission Delay Exceeded Deadline
                               print('job EXCEEDING DEADLINE')
                               penalty_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                               rejected_jobs_deadline_t[t_slot] += 1

                    total_jobs += 1
                    revenueTotalJobs_t.append(env.REV_FACTOR * env.c_jn * jobList_aux[j])
                    num_newArrivedjobs[t_slot] += 1

            # ---------------------------------------------------------
            # RECORD POWER & COST METRICS FOR THIS TIMESLOT
            # ---------------------------------------------------------

            nodesPW = env.getPowerConsumption()

            revenue_j[t_slot]             = revenueAcceptedJobs_t
            total_revenue_j[t_slot]       = revenueTotalJobs_t
            penalty[t_slot]               = penalty_t

            # Record job acceptance and rejection statistics
            accepted_jobs_t[t_slot] = np.sum(isAccepted)
            rejection_own[t_slot]   = rejected_num_jobs ## rejected because of fair decision
            rejected_jobs_t[t_slot] = rejected_num_jobs + rejected_jobs_power_t[t_slot] + rejected_jobs_deadline_t[t_slot]

            capacity_violation_rejected_new_jobs_t[t_slot]    = rejected_jobs_power_t[t_slot]
            deadline_violation_rejected_new_jobs_t[t_slot]    = rejected_jobs_deadline_t[t_slot]


            interrupted_jobs_wrong_decision_t[t_slot]         = mig_reject_violation # = 0
            interrupted_jobs_power_t[t_slot]                  = mig_power_violation # = 0
            interrupted_jobs_delay_t[t_slot]                  = mig_delay_violation

            migration_violation_interrupted_jobs_t[t_slot]    = mig_reject_violation # = 0
            capacity_violation_mig_interrupted_jobs_t[t_slot] = mig_power_violation # = 0
            deadline_violation_mig_interrupted_jobs_t[t_slot] = mig_delay_violation # = 0

            # Calculate brown (non-renewable) power usage per server
            for nn in range(env.N_SERVERS):
                brown_power = max(nodesPW[nn] - env.greenPWrealization[nn], 0)
                P_nt[t_slot, nn]         = brown_power
                pw_nt_sol[nn, t_slot]    = nodesPW[nn]
                pw_nt_br_sol[nn, t_slot] = brown_power
                pw_nt_gr_sol[nn, t_slot] = env.greenPWrealization[nn]

        # ---------------------------------------------------------
        # AGGREGATE RESULTS FOR THE REALIZATION
        # ---------------------------------------------------------

        results_eval_n ={}
        results_eval_n['pw_nt_sol']    = pw_nt_sol
        results_eval_n['pw_nt_br_sol'] = pw_nt_br_sol
        results_eval_n['pw_nt_gr_sol'] = pw_nt_gr_sol
        results_eval_n['cost_br_pw']   = np.sum(pw_nt_br_sol,axis =0)*env.COST_FACTOR


        results_eval_n['total_brown_used']      = np.sum(pw_nt_br_sol)
        results_eval_n['total_green_available'] = np.sum(pw_nt_gr_sol)
        results_eval_n['total_green_not_used']  = np.sum(pw_nt_gr_sol) - (np.sum(pw_nt_sol) - np.sum(pw_nt_br_sol))
        results_eval_n['accepted_jobs_t']       = accepted_jobs_t
        results_eval_n['rejected_jobs_t']       = rejected_jobs_t
        results_eval_n['rejected_jobs_power_t'] = rejected_jobs_power_t
        results_eval_n['rejected_jobs_delay_t'] = rejected_jobs_deadline_t
        results_eval_n['rejection_own']         = rejection_own # jobs for which algorithm decides not to  accept them -- not due to errors 

        results_eval_n['interrupted_rejected_t'] = interrupted_jobs_wrong_decision_t # rejections when migrating # NOT USED # = 0
        results_eval_n['interruptedpower_t']     = interrupted_jobs_power_t # rejecting due to migration to full node # NOT USED # = 0
        results_eval_n['interruptedelay_t']      = interrupted_jobs_delay_t # rejecting due to migration to full node

        results_eval_n['n_migrations_t']        = np.sum(n_migrations_t)
        results_eval_n['n_migrations_t_total']  = np.mean(n_migrations_t) # = 0

        results_eval_n['job_pc_list']       = jobs_pc
        results_eval_n['job_dur_list']      = jobs_duration
        results_eval_n['job_delay_list']    = jobs_wdlay
        results_eval_n['propagation_delay'] = env.matrix_propagation_delay

        results_eval_n['migration_violation_interrupted_jobs']    = np.sum(migration_violation_interrupted_jobs_t)
        results_eval_n['capacity_violation_rejected_new_jobs']    = np.sum(capacity_violation_rejected_new_jobs_t)
        results_eval_n['capacity_violation_mig_interrupted_jobs'] = np.sum(capacity_violation_mig_interrupted_jobs_t)
        results_eval_n['deadline_violation_rejected_new_jobs']    = np.sum(deadline_violation_rejected_new_jobs_t)

        # Compute moving average KPIs and overall final metrics (excluding transient period)
        results_eval_n = getResults(results_eval_n, env, P_nt, pw_nt_gr_sol, tot_time_evaluation, window_size, num_newArrivedjobs, accepted_jobs_t, total_jobs,  revenue_j, penalty, total_revenue_j, TRANSIENT, NOLOG_FAIRNESS)
        results_eval = append_(results_eval, results_eval_n)

    return results_eval


