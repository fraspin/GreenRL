import os
import pickle
import numpy as np
import argparse
import yaml

##### Stable Baselines3 Imports
# Used for standard RL algorithms (DQN, A2C) and environment wrappers/utilities

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, ProgressBarCallback
from stable_baselines3 import DQN, A2C
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env


#### Local Classes and Functions Imports
# Core simulator files containing environments, evaluation scripts, and plotting

from auxiliarFunctions import *
from environ import WorldEnv
from environ_sota import WorldEnvStateOfTheArt
from evaluation_algorithms  import evaluate_algorithm
from evaluation_sota import evaluate_sota
from Solver import Solver


parser = argparse.ArgumentParser(description="Load data from a yaml file.")
parser.add_argument('config_yaml', type=str, help='Path to the configuration YAML file.')
parser.add_argument('params_yaml', type=str, help='Path to the experiment parameters YAML file.')
args = parser.parse_args()

config_data = load_yaml_configuration(args.config_yaml)
data_experiment = load_yaml_configuration(args.params_yaml)

# ============================================================================
# SIMULATION STAGE FLAGS
# Toggle these flags (1 to enable, 0 to disable) to choose which parts
# of the pipeline to run.
# ============================================================================
GREENRL_TRAINING      = config_data['GREENRL_TRAINING']
SOTA_TRAINING         = config_data['SOTA_TRAINING']
POLICY_EVALUATION     = config_data['POLICY_EVALUATION']
SOTA_EVALUATION       = config_data['SOTA_EVALUATION']
BASELINES_EVALUATION  = config_data['BASELINES_EVALUATION']
SOLVER_EVALUATION     = config_data['SOLVER_EVALUATION']
CARBONEDGE_EVALUATION = config_data['CARBONEDGE_EVALUATION']

# ============================================================================
# GLOBAL SETTINGS
# ============================================================================
REVCOST_F            = config_data['REVCOST_F']
CONTINUAL_LEARNING   = config_data['CONTINUAL_LEARNING']
PRINT_FIGURES        = config_data['PRINT_FIGURES']


# ============================================================================
# CONFIGURATION LOADING
# Retrieve parameters for the overall system simulation from a provided YAML file.
# ============================================================================
# Define directories to save and load experiments
experimentsFolder   = "test_experiments"
exp_greenRL         = data_experiment['EXP_GREENRL']
exp_sota            = data_experiment['EXP_SOTA']
experiment_dir      = os.path.join(experimentsFolder, exp_greenRL)
experiment_dir_sota = os.path.join(experimentsFolder, exp_sota)


# Model filenames mapping
# 'best_model' relies on EvalCallback saving the best performing model.
last_model_GreenRL = "Last_training_model_GreenRL"
last_model_SOTA    = "Last_training_model_SOTA"
best_model_GreenRL = "best_model"
best_model_SOTA    = "best_model"
dir_last_exp_name_GREENRL = 'name_lastest_experiment_GREENRL.pkl'
dir_last_exp_name_SOTA    = 'name_lastest_experiment_SOTA.pkl'


# Automatically load the latest experiment directory if 'latest' is passed and we aren't training
if experiment_dir == 'latest' and not GREENRL_TRAINING:
    with open(dir_last_exp_name_GREENRL, 'rb') as mfile:
        experiment_dir = pickle.load(mfile)
        
if experiment_dir_sota == 'latest' and not SOTA_TRAINING:
    with open(dir_last_exp_name_SOTA, 'rb') as mfile:
        experiment_dir_sota = pickle.load(mfile)

environmentOfTraining = 0

# ============================================================================
# ENVIRONMENT & SIMULATION PARAMETERS EXTRACTION
# These parameters map directly from the loaded YAML configuration.
# ============================================================================

# Nodes parameters (Edge Servers infrastructure)
N_SERVERS              = data_experiment['N_SERVERS']
MAX_COMPUTATION_SERVER = data_experiment['MAX_COMPUTATION_SERVER_NORMALIZED']
MAX_POWER_SERVER       = data_experiment['MAX_POWER_SERVER_NORMALIZED']
MAX_STORAGE_SERVER     = data_experiment['MAX_STORAGE_SERVER']                 # 1 TB
TERAFLOPS_SERVER       = data_experiment['TERAFLOPS_SERVER']

# Jobs/Tasks parameters (Workload characterization)
ARRIVAL_RATE       = data_experiment['ARRIVAL_RATE']
JOB_SESSION_LENGTH = data_experiment['JOB_SESSION_LENGTH']
JOB_STORAGE_NEED   = data_experiment['JOB_STORAGE_NEED']       # MB
JOB_CYCLES_TS      = data_experiment['JOB_CYCLES_NORMALIZED']
JOB_TERAFLOPS      = data_experiment['JOB_TERAFLOPS']
FRAME_SIZE         = data_experiment['FRAME_SIZE']

# Network parameters (Communication capabilities)
MAX_DATARATE     = data_experiment['MAX_DATARATE']
MIN_DATARATE     = data_experiment['MIN_DATARATE']
FIBER_DATARATE   = data_experiment['FIBER_DATARATE']
MIGRATION_DUR_TS = data_experiment['MIGRATION_DUR_TS']
MAX_USERSxNODE   = data_experiment['MAX_USERSxNODE']

# Delay parameters (Constraints)
DEADLINE_DELAY    = data_experiment['FPS_DEADLINE_DELAY']
SWITCHING_DELAY   = data_experiment['SWITCHING_DELAY']
COMPUTATION_DELAY = (MAX_USERSxNODE * JOB_TERAFLOPS) / TERAFLOPS_SERVER * 1000  # ms

# Revenue and costs parameters (Financial modeling)
REVENUE_FACTOR = data_experiment['REVENUE_FACTOR']
COST_FACTOR    = data_experiment['COST_FACTOR']

# Geometry parameters (Spatial distribution)
SCENARIO_KM_SIDE = data_experiment['SCENARIO_KM_SIDE']  # Size of the square where edge nodes are located

# Objective Functions parameters (Weights for Multi-Objective Formulation)
alpha      = data_experiment['alpha']
beta       = data_experiment['beta']
gamma      = data_experiment['gamma']
rho_d      = data_experiment['rho_d']
rho_p      = data_experiment['rho_p']
rho_r      = data_experiment['rho_r']
pw_params  = (alpha, beta, gamma)
obj_params = (rho_d, rho_p, rho_r)

# Overall simulation control parameters
tot_time_evaluation    = data_experiment['tot_time_evaluation']
n_real_eval            = data_experiment['n_real_eval']
training_window_size   = data_experiment['training_window_size']
evaluating_window_size = data_experiment['evaluating_window_size']
TRANSIENT              = int(tot_time_evaluation * \
                             data_experiment.get('TRANSIENT_PERCT_EVAL', 0)
                            )
NOLOG_FAIRNESS         = data_experiment.get('NOLOG_FAIRNESS', 0)
MAX_STEPS_EPISODE      = data_experiment.get('MAX_STEPS_EPISODE', 1024)


# SOLVER parameters (For the exact/heuristic optimization benchmark)
time_opt_solver      = data_experiment['time_opt_solver']       # window for which the solver runs the optimization
talkative            = data_experiment['talkative']             # More printing during optimization
time4solver          = data_experiment['time4solver']           # Max. time for each opt. iteration to finish
numMaxIter           = data_experiment['numMaxIter']            # Max. number of iterations of optimization
gapPostDeadline      = data_experiment['gapPostDeadline']       # % Gap w.r.t. optimal for next interation stop
gapIncremStop        = data_experiment['gapIncremStop']         # keep it 0 for now. It can give bad gap solutions
max_extension_solver = data_experiment['max_extension_solver']  # Extend time slots for solver convergence / conclusion



# ============================================================================
# AGENT TRAINING CONFIGURATIONS
# ============================================================================

# Load specific hyperparameters if training the proposed GreenRL agent (A2C)
if GREENRL_TRAINING == 1:

    yaml_file_path = 'training_GREENRL_parameters.yml'
    config_data = load_yaml_configuration(yaml_file_path)

    N_parallelEnvs  = config_data['N_parallelEnvs']
    training_params = config_data['training_parameters']

    NUM_TOT_timesteps    = data_experiment['NUM_TOT_timesteps']     # data from previous yaml file since training time is proportional to scenario size

    learningRate    = training_params['learningRate']
    N_epochs        = training_params['N_epochs']
    N_STEPS_model   = training_params['N_STEPS_model']
    bacthSize       = training_params['bacthSize']
    
    MAX_STEPS_EPISODE    = training_params['MAX_STEPS_EPISODE']
    freq_check_bestTrain = training_params['freq_check_bestTrain']
    freq_eval_callBack   = training_params['freq_eval_callBack']
    max_eval_episodes    = training_params['max_eval_episodes']
    gae_lambda_val       = training_params['gae_lambda_val']
    gamma_val            = training_params['gamma_val']
    clip_range_val       = training_params['clip_range_val']
    clip_range_vf        = training_params['clip_range_vf']
    ent_coef             = training_params['ent_coef']
    big_penalty          = training_params['big_penalty']
    policy_kwargs        = config_data['policy_kwargs']
    N_parallelEnvs       = 16

# Load specific hyperparameters if training the SOTA agent (CC23). It uses DQN
if SOTA_TRAINING == 1:
    
    yaml_file_path = 'training_SOTA_parameters.yml'
    config_data = load_yaml_configuration(yaml_file_path)

    learningRate_sota         = config_data['learningRate_sota']
    gamma_val_sota            = config_data['gamma_val_sota']
    exploration_initial_eps   = config_data['exploration_initial_eps']
    exploration_final_eps     = config_data['exploration_final_eps']
    N_parallelEnvs            = 8


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def model_learning(envd, experiment_dir, environ_params, GreenRl_training, last_model, best_model):
    """
        Sets up the RL environment, defines the model algorithm (A2C or DQN), and executes the training loop.

        Args:
            envd: Base environment instance (used to extract definitions/spaces).
            experiment_dir: Directory where logs and models will be saved.
            environ_params: Dictionary of parameters to initialize the vectorized environments.
            GreenRl_training: Flag (1 or 0) indicating whether we are training GreenRL (A2C) or SOTA (DQN).
            last_model: Filename to save the final state of the model.
            best_model: Filename to save the best performing model via callbacks.

        Returns:
            model: The fully trained Stable-Baselines3 model.
            env: The vectorized environment used during training.
        """
    max_eval_episodes = 10
    freq_eval_callBack = 100000

    env = envd
    check_env(env, warn=True) # Validates that the custom environment follows the Gym API
    # Callback to periodically evaluate the model and save the best one
    eval_callback = EvalCallback(eval_env             = env,
                                 callback_on_new_best = None,
                                 n_eval_episodes      = max_eval_episodes,
                                 eval_freq            = freq_eval_callBack,
                                 best_model_save_path = experiment_dir,
                                 # best_model_save_name = best_model,
                                 log_path             = os.path.join(experiment_dir,'logs'),
                                 deterministic        = False,
                                 render               = False,
                                 verbose              = 1)
    progess_callback = ProgressBarCallback()
 
    # Select the appropriate environment class based on the agent type
    if GreenRl_training == 1:
        myEnv = WorldEnv
    else:
        myEnv = WorldEnvStateOfTheArt
    # Create multiple parallel environments to speed up data collection
    N_parallelEnvs = 8
    env1 = make_vec_env(myEnv, 
                        n_envs      = N_parallelEnvs,
                        monitor_dir = experiment_dir, 
                        env_kwargs  = environ_params)
    

    # Initialize the learning algorithm or load an existing one
    if CONTINUAL_LEARNING == 0:
        if GreenRl_training == 1:
            # Initialize A2C for GreenRL
            model = A2C("MultiInputPolicy", env1, learning_rate=learningRate,
                        n_steps=N_STEPS_model, gamma=gamma_val, gae_lambda=gae_lambda_val,
                        ent_coef=ent_coef, vf_coef=0.2, max_grad_norm=0.5, rms_prop_eps=1e-05,
                        use_rms_prop=True, use_sde=False, sde_sample_freq=4, normalize_advantage=False,
                        tensorboard_log="logs/A2C", policy_kwargs=policy_kwargs, verbose=0, seed=None,
                        device='auto', _init_setup_model=True)

        else:
            # Initialize DQN for the SOTA agent (CC23)
            model = DQN("MultiInputPolicy", env1,learning_rate=learningRate_sota,gamma=gamma_val_sota,
                        exploration_initial_eps = exploration_initial_eps, exploration_final_eps=exploration_final_eps)

    else:
        # Continual learning: load an existing model to resume training
        if GreenRl_training == 1:
            model = A2C.load(path.join(experiment_dir, last_model_GreenRL), env=env1)
        else:
            model = DQN.load(path.join(experiment_dir, last_model_SOTA), env=env1)

    print("\n+-+-+-+-+\n starting learning \n+-+-+-+-+\n")

    # Start the training process
    model.learn(total_timesteps=NUM_TOT_timesteps, callback=[progess_callback, eval_callback])

    # Save the final model state post-training
    model.save(path.join(experiment_dir, last_model))  ### Saving model

    return (model, env)


def trainAgent(exp_dir, experiment_dir_cont, dir_last_experiment_name, lastAndBestModelNames, continuingLearning, isGreenRL, doPlots):
    """
        Wrapper function to orchestrate the pre-training setup, start the learning process,
        and handle post-training serialization and plotting.

        Args:
            exp_dir: Directory where the new experiment logs and models will be saved.
            experiment_dir_cont: Directory to load from if continuing from a previous run.
            dir_last_experiment_name: Path to save the name of the current experiment (for later retrieval).
            lastAndBestModelNames: Tuple containing (last_model_filename, best_model_filename).
            continuingLearning: Flag indicating if we are loading an old model.
            isGreenRL: 1 if we are training the GreenRL agent, 0 for SOTA.
            doPlots: 1 to generate training curve plots after completion, 0 otherwise.
        """

    last_model_name, best_model_name = lastAndBestModelNames

    # Persist the current experiment directory name so future evaluations know where to look
    with open(dir_last_experiment_name, 'wb') as mfile:
        pickle.dump(exp_dir, mfile)
        
    name_pkl = 'saved_environment_GreenRL.pkl' if isGreenRL else 'saved_environment_sota.pkl'
    
    myEnv    = WorldEnv if isGreenRL else WorldEnvStateOfTheArt

    # Environment initialization or loading
    if CONTINUAL_LEARNING == 0:
        experiment_dir   = exp_dir
        envvv            = myEnv(**env_params)
        envvv            = TimeLimit(envvv, max_episode_steps=MAX_STEPS_EPISODE)
        environment_file = os.path.join(experiment_dir, name_pkl)
    else:
        experiment_dir   = experiment_dir_cont
        environment_file = os.path.join(experiment_dir, name_pkl)
        # Load previous environment state to maintain continuity
        with open(environment_file, 'rb') as mfile:
            envvv = pickle.load(mfile)

    os.makedirs(experiment_dir, exist_ok=True)

    # Trigger the core learning loop
    model, _ = model_learning(envvv, experiment_dir, env_params, 
                              isGreenRL, last_model_name, best_model_name)  

    envvv.reset()

    # Save the environment state post-training
    with open(environment_file, 'wb') as mfile:
        pickle.dump(envvv, mfile)

    # Post-training visualization
    figure_directory = os.path.join(experiment_dir, 'figures')
    os.makedirs(figure_directory, exist_ok=True)
    os.makedirs(os.path.join(figure_directory, 'bar_plots'), exist_ok=True)


# Dictionary dictating which components to evaluate
EVALUATION_SETUP = {"POLICY":     POLICY_EVALUATION,
                    "SOTA":       SOTA_EVALUATION,
                    "BASELINES":  BASELINES_EVALUATION,
                    "CARBONEDGE": CARBONEDGE_EVALUATION,
                    "SOLVER":     SOLVER_EVALUATION
                   }

if __name__ == "__main__":
    """
    Main entry point. Initializes parameters, triggers training phases if requested, 
    and handles all evaluation, statistics generation, and final plotting.
    """

    # Core dictionary holding all variables required to instantiate the simulated environment.
    # Shared between both the GreenRL and SOTA (CC23) setups.
    env_params = dict(
         num_servers        = N_SERVERS,
         max_pc_server      = MAX_COMPUTATION_SERVER,
         arrivalRate        = ARRIVAL_RATE,
         max_pw_server      = MAX_POWER_SERVER,
         jobSessionLength   = JOB_SESSION_LENGTH,
         pw_params          = pw_params,
         obj_params         = obj_params,
         revenue_factor     = REVENUE_FACTOR,
         cost_factor        = COST_FACTOR,
         c_jn               = JOB_CYCLES_TS,
         computationDelay   = COMPUTATION_DELAY,
         switchingDelay     = SWITCHING_DELAY,
         deadlineDelay      = DEADLINE_DELAY,
         minWirelessRate    = MIN_DATARATE,
         maxWirelessRate    = MAX_DATARATE,
         fiberDataRate      = FIBER_DATARATE,
         frameSize          = FRAME_SIZE,
         maxStorageServer   = MAX_STORAGE_SERVER,
         job_storage        = JOB_STORAGE_NEED,
         T_buffer           = training_window_size,
         window_m           = MIGRATION_DUR_TS,
         revCost_function   = REVCOST_F,
         nolog_fairness     = NOLOG_FAIRNESS,
         setting_size       = SCENARIO_KM_SIDE
     )

    # ------------------------------------------------------------------------
    # TRAINING PHASES
    # ------------------------------------------------------------------------
    log_dir = folderPathGenerator(experimentsFolder)

    if GREENRL_TRAINING == 1:
        log_dir_policy = log_dir + "_policy"

        trainAgent(log_dir_policy,             ## Directory to save results
                   experiment_dir,             ## in case of continuous learning
                   dir_last_exp_name_GREENRL,  ## To save current experiment
                   [last_model_GreenRL, best_model_GreenRL], ## names of files for last_model and best_model
                   CONTINUAL_LEARNING,         ## 1 if training continues from previous models
                   isGreenRL = 1,              ## 1 is GreenRL, 0 if SOTA
                   doPlots = PRINT_FIGURES     ## If you want figures about training
                 )

        if CONTINUAL_LEARNING == 0:
            experiment_dir = log_dir_policy


    if SOTA_TRAINING == 1:
        log_dir_sota = log_dir + "_sota"
        trainAgent(log_dir_sota,               ## Directory to save results
                   experiment_dir_sota,        ## in case of continuous learning
                   dir_last_exp_name_SOTA,     ## To save current experiment
                   [last_model_SOTA, best_model_SOTA], ## names of files for last_model and best_model
                   CONTINUAL_LEARNING,         ## 1 if training continues from previous models
                   isGreenRL = 0,              ## 1 is GreenRL, 0 if SOTA
                   doPlots = PRINT_FIGURES     ## If you want figures about training
                 )

        if CONTINUAL_LEARNING == 0:
            experiment_dir_sota = log_dir_sota

    # ------------------------------------------------------------------------
    # EVALUATION PHASES
    # Evaluates trained models, baselines, and solvers over 'n_real_eval' realizations.
    # ------------------------------------------------------------------------

    ## First, check what algorithms have been evaluated already.
    ## keep them if they are not overwritten algorithms follow a strict order.
    # Evaluation_stage = 1 is GreenRL evaluated, 2 if baselines evaluated, 3 if sota evaluated, 4 if carbonedge.
    results_eval_file   = os.path.join(experiment_dir, 'results_eval.pkl')
    try:
        backup, eval_status = read_pickle_objects(results_eval_file, EVALUATION_SETUP)
    except FileNotFoundError:
        backup = {}
        eval_status = {"final_stage": 0}

    # 1. Evaluate the proposed Policy (GreenRL)
    if POLICY_EVALUATION == 1:
        # Align directories if trained in this exact run
        if GREENRL_TRAINING == 1:
            if CONTINUAL_LEARNING == 0:
                pass
        else:
            pass

        # Create or restore the specific environment instance
        if environmentOfTraining == 1:
            environment_file = os.path.join(experiment_dir, 'saved_environment_GreenRL.pkl')
            if os.path.isfile(environment_file):
                with open(environment_file, 'rb') as mfile:
                    envx = pickle.load(mfile)
            envx.reset()
            envx.jobSessionLength = JOB_SESSION_LENGTH
        else:
            envx = WorldEnv(**env_params)

        print('\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\nEVALUATING POLICY',
              '\n+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n')

        # Load the model and perform inference across the evaluation timesteps
        model = A2C.load(path.join(experiment_dir, last_model_GreenRL), env=envx)

        ### Evaluation of the trained model
        results_eval = evaluate_algorithm('policy', envx, data_experiment, results_policy=None, model=model)

        backup["results_eval"] = results_eval
        backup["env"] = envx

    # 2. Evaluate simplistic baselines against the exact same scenario environment
    if BASELINES_EVALUATION == 1:
        results_eval_random    = evaluate_algorithm('random',    backup["env"], data_experiment, backup["results_eval"])
        results_eval_emptier   = evaluate_algorithm('emptier',   backup["env"], data_experiment, backup["results_eval"])
        results_eval_heuristic = evaluate_algorithm('heuristic', backup["env"], data_experiment, backup["results_eval"])
        backup["results_eval_random"]    = results_eval_random     
        backup["results_eval_emptier"]   = results_eval_emptier    
        backup["results_eval_heuristic"] = results_eval_heuristic

    # 3. Evaluate the State-of-the-Art (SOTA) agent (CC23)
    if SOTA_EVALUATION == 1:
        if SOTA_TRAINING == 1 and CONTINUAL_LEARNING == 0:
            experiment_dir_sota = log_dir
        if environmentOfTraining == 1:
            environment_file = os.path.join(experiment_dir_sota, 'saved_environment_sota.pkl')
            if os.path.isfile(environment_file):
                with open(environment_file, 'rb') as mfile:
                    envx = pickle.load(mfile)
            envx.reset()
            envx.jobSessionLength = JOB_SESSION_LENGTH
        else:
            envx = WorldEnvStateOfTheArt(**env_params)

        model                  = DQN.load(path.join(experiment_dir_sota, last_model_SOTA), env=envx)
        results_eval_state_art = evaluate_sota(envx, data_experiment, backup["results_eval"], model)
        backup["results_eval_sota"]    = results_eval_state_art

        # 4. Evaluate the CarbonEdge benchmark
    if CARBONEDGE_EVALUATION == 1:                
        results_eval_carbonedge = evaluate_algorithm('carbonedge', backup["env"], data_experiment, backup["results_eval"])
        backup["results_eval_carbonedge"] = results_eval_carbonedge

    with open(results_eval_file, 'wb') as mfile:

        if "results_eval" in backup and "env" in backup:
            pickle.dump(backup["results_eval"], mfile)
            pickle.dump(backup["env"], mfile)
        if "results_eval_random" in backup:
            pickle.dump(backup["results_eval_random"], mfile)
            pickle.dump(backup["results_eval_emptier"], mfile)
            pickle.dump(backup["results_eval_heuristic"], mfile)
        if "results_eval_sota" in backup:
            pickle.dump(backup["results_eval_sota"], mfile)
        if "results_eval_carbonedge" in backup:
            pickle.dump(backup["results_eval_carbonedge"], mfile)

    # 5. Evaluate the numerical solver (Upper bound/Exact Optimization)
    if SOLVER_EVALUATION == 1:
        results_solver_file = os.path.join(experiment_dir, 'results_solver.pkl')

        # Retrieve the environment history so the solver runs on the exact same workload
        with open(results_eval_file, 'rb') as mfile:
            results_eval = pickle.load(mfile)
            env = pickle.load(mfile)

        results_solver       = []
        revRatioTotal_solver = []

        # Iterate through the number of simulation runs (realizations)
        for realization in range(n_real_eval):
            # Extract state data directly from the policy evaluation logs
            jobs     = results_eval[realization]['job_pc_list']
            jobs_dur = results_eval[realization]['job_dur_list']
            greenPW  = results_eval[realization]['pw_nt_gr_sol']

            jobs_wd   = results_eval[realization]['job_wdlay_list'] ### wireless delay (oneway)
            nodes_pd  = results_eval[realization]['propagation_delay'] ### propagation delay (oneway)

            timesteps = len(jobs)

            print('\n', '\n+-+-+-+-+-+-+-+-+-+-+' * 2,
                  '\n', '\n+-+-+-+-+-+-+-++-+-+-+-+-+-+-++-+-+-+-+-+-+-+' * 3,
                  f'\n     SOLVER EVALUATION NUMBER    {realization + 1}/{n_real_eval} ',
                  '\n+-+-+-+-+-+-+-++-+-+-+-+-+-+-++-+-+-+-+-+-+-+' * 3,
                  '\n', '\n+-+-+-+-+-+-+-+-+-+-+' * 2, '\n')

            print('JOBS:\n', jobs)
            print('GREEN ENERGY:\n', greenPW)
            print('Wireless DELAY:\n', jobs_wd)
            print('E2E propagation DELAY:\n', nodes_pd)

            mean_duration = int(np.array([job for jobs_t in jobs_dur for job in jobs_t]).mean())
            extended_time_solver = min(mean_duration, max_extension_solver)

            # Initialize the Solver with a look-ahead window
            mySolver = Solver(env, time_opt_solver, timesteps, margin_last_jobs=extended_time_solver,
                              deadline_solver=time4solver, num_Max_iter=numMaxIter,
                              gapPostdeadline=gapPostDeadline, gapIncrementStop=gapIncremStop, revcost_function=REVCOST_F,
                              nologFair=NOLOG_FAIRNESS)


            myTimer = TimeCounter(1)
            history = []
            myTimer.setTimeOn()

            instanceNodesPD = nodes_pd

            # Run optimization slot by slot
            for ttt in range(mySolver.T_tot):
                print(f'\n+-+-+-+-+-+-+-+\nRunning Time Slot {ttt}\n+-+-+-+-+-+-+-+\n')
                # Grab a window of future data (receding horizon style)
                if ttt <= mySolver.T_tot - time_opt_solver:
                    instanceJobs    = jobs[ttt: ttt + time_opt_solver]
                    instanceJobsDur = jobs_dur[ttt: ttt + time_opt_solver]
                    instanceJobsWD  = jobs_wd[ttt: ttt + time_opt_solver]
                    instanceGPWs    = greenPW[:, ttt: ttt + time_opt_solver]
                else:
                    # Pad edges with empty jobs for the remaining slots
                    instanceJobs    = jobs[ttt:]     + [[]] * (time_opt_solver - (mySolver.T_tot - ttt))
                    instanceJobsDur = jobs_dur[ttt:] + [[]] * (time_opt_solver - (mySolver.T_tot - ttt))
                    instanceJobsWD  = jobs_wd[ttt:]  + [[]] * (time_opt_solver - (mySolver.T_tot - ttt))
                    instanceGPWs    = np.concatenate(
                        (greenPW[:, ttt:], np.repeat(greenPW[:, -1:],
                                                     [time_opt_solver - (mySolver.T_tot - ttt)], axis=1)),
                        axis=1)

                # Construct the LP/ILP problem and solve it
                optProb        = mySolver.createInstance(instanceJobs, instanceJobsDur, instanceJobsWD, instanceNodesPD, instanceGPWs)
                instanceOutput = optProb.solveInstance(talkative)
                history.append(instanceOutput)
                myTimer.offAndOn('\nIteration ' + str(ttt) + ' -- ')
            myTimer.getTimeOff()

            # Compile solver results for comparison with RL/heuristics
            results_solver_i = mySolver.createPlottingStats(greenPW, env.COST_FACTOR, evaluating_window_size)
            results_solver   = append_(results_solver, results_solver_i)

        # Save solver results
        with open(results_solver_file, 'wb') as mfile:
            pickle.dump(results_solver, mfile)


# ============================================================================
# STATISTICS & METRICS COMPUTATION
# ============================================================================

print('\n\n\n/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/ AVERAGE VALUES /+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/')

# Aggregate the logged data across all evaluation runs (realizations)
if POLICY_EVALUATION == 1:
    revRatioTotal = [results_eval[n_time]['RevenueRatioTotal']    for n_time in range(n_real_eval)]
    ObjF_RevCost  = [results_eval[n_time]['Objective_function_rev_minus_cost'] for n_time in range(n_real_eval)]
    ObjFFairness  = [results_eval[n_time]['objFunction_Fairness'] for n_time in range(n_real_eval)]
    PowerTotal    = [results_eval[n_time]['normPower_fair']       for n_time in range(n_real_eval)]
    revFairness   = [results_eval[n_time]['normRev_fair']         for n_time in range(n_real_eval)]
    mean_mig      = np.mean([results_eval[n_time]['n_migrations_t'] for n_time in range(n_real_eval)])
    revenueTotal  = [results_eval[n_time]['revenueTotal']         for n_time in range(n_real_eval)]
    costPWTotal   = [results_eval[n_time]['costPWTotal']          for n_time in range(n_real_eval)]
    penaltyTotal  = [results_eval[n_time]['penaltyTotal']         for n_time in range(n_real_eval)]


    Migration_violation_interrupted_jobs    = [results_eval[n_time]['migration_violation_interrupted_jobs'] for n_time in
                                            range(n_real_eval)]
    Capacity_violation_rejected_new_jobs    = [results_eval[n_time]['capacity_violation_rejected_new_jobs'] for n_time in
                                            range(n_real_eval)]
    Capacity_violation_mig_interrupted_jobs = [results_eval[n_time]['capacity_violation_mig_interrupted_jobs'] for
                                               n_time in range(n_real_eval)]

if SOTA_EVALUATION == 1:

    revRatioTotal_sota = [results_eval_state_art[n_time]['RevenueRatioTotal'] for n_time in range(n_real_eval)]
    ObjF_RevCost_sota  = [results_eval_state_art[n_time]['Objective_function_rev_minus_cost'] for n_time in
                            range(n_real_eval)]
    ObjFFairness_sota   = [results_eval_state_art[n_time]['objFunction_Fairness'] for n_time in range(n_real_eval)]
    PowerTotal_sota     = [results_eval_state_art[n_time]['normPower_fair'] for n_time in range(n_real_eval)]
    revFairness_sota    = [results_eval_state_art[n_time]['normRev_fair'] for n_time in range(n_real_eval)]
    revenueTotal_sota   = [results_eval_state_art[n_time]['revenueTotal'] for n_time in range(n_real_eval)]
    costPWTotal_sota    = [results_eval_state_art[n_time]['costPWTotal'] for n_time in range(n_real_eval)]
    penaltyTotal_sota   = [results_eval_state_art[n_time]['penaltyTotal'] for n_time in range(n_real_eval)]
    mean_migration_sota = np.mean([results_eval_state_art[n_time]['n_migrations_t'] for n_time in range(n_real_eval)])
else:
    results_eval_state_art = []

if CARBONEDGE_EVALUATION    == 1:
   revRatioTotal_carbonedge  = [results_eval_carbonedge[n_time]['RevenueRatioTotal'] for n_time in range(n_real_eval)]
   ObjF_RevCost_carbonedge   = [results_eval_carbonedge[n_time]['Objective_function_rev_minus_cost'] for n_time in range(n_real_eval)]
   ObjFFairness_carbonedge   = [results_eval_carbonedge[n_time]['objFunction_Fairness'] for n_time in range(n_real_eval)]
   PowerTotal_carbonedge     = [results_eval_carbonedge[n_time]['normPower_fair'] for n_time in range(n_real_eval)]
   revFairness_carbonedge    = [results_eval_carbonedge[n_time]['normRev_fair'] for n_time in range(n_real_eval)]
   revenueTotal_carbonedge   = [results_eval_carbonedge[n_time]['revenueTotal'] for n_time in range(n_real_eval)]
   costPWTotal_carbonedge    = [results_eval_carbonedge[n_time]['costPWTotal'] for n_time in range(n_real_eval)]
   penaltyTotal_carbonedge   = [results_eval_carbonedge[n_time]['penaltyTotal'] for n_time in range(n_real_eval)]
   carbonedge_emissions      = [results_eval_carbonedge[n_time]['carbonedge_total_emissions'] for n_time in range(n_real_eval)]
else:
    results_eval_carbonedge = []

if BASELINES_EVALUATION == 1:

    revRatioTotal_random     = [results_eval_random[n_time]['RevenueRatioTotal'] for n_time in range(n_real_eval)]
    ObjF_RevCost_random      = [results_eval_random[n_time]['Objective_function_rev_minus_cost'] for n_time in range(n_real_eval)]
    ObjFFairness_random      = [results_eval_random[n_time]['objFunction_Fairness'] for n_time in range(n_real_eval)]
    PowerTotal_random        = [results_eval_random[n_time]['normPower_fair'] for n_time in range(n_real_eval)]
    revFairness_random       = [results_eval_random[n_time]['normRev_fair'] for n_time in range(n_real_eval)]
    revenueTotal_random      = [results_eval_random[n_time]['revenueTotal'] for n_time in range(n_real_eval)]
    costPWTotal_random       = [results_eval_random[n_time]['costPWTotal'] for n_time in range(n_real_eval)] 

    revRatioTotal_heuristic  = [results_eval_heuristic[n_time]['RevenueRatioTotal'] for n_time in range(n_real_eval)]
    ObjF_RevCost_heuristic   = [results_eval_heuristic[n_time]['Objective_function_rev_minus_cost'] for n_time in range(n_real_eval)]
    ObjFFairness_heuristic   = [results_eval_heuristic[n_time]['objFunction_Fairness'] for n_time in range(n_real_eval)]
    PowerTotal_heuristic     = [results_eval_heuristic[n_time]['normPower_fair'] for n_time in range(n_real_eval)]
    revFairness_heuristic    = [results_eval_heuristic[n_time]['normRev_fair'] for n_time in range(n_real_eval)]
    mean_migration_heuristic = np.mean([results_eval_heuristic[n_time]['n_migrations_t'] for n_time in range(n_real_eval)])
    revenueTotal_heuristic   = [results_eval_heuristic[n_time]['revenueTotal'] for n_time in range(n_real_eval)]
    costPWTotal_heuristic    = [results_eval_heuristic[n_time]['costPWTotal'] for n_time in range(n_real_eval)]
    penaltyTotal_heuristic   = [results_eval_heuristic[n_time]['penaltyTotal'] for n_time in range(n_real_eval)] 

    revRatioTotal_emptier    = [results_eval_emptier[n_time]['RevenueRatioTotal'] for n_time in range(n_real_eval)]
    ObjF_RevCost_emptier     = [results_eval_emptier[n_time]['Objective_function_rev_minus_cost'] for n_time in range(n_real_eval)]
    ObjFFairness_emptier     = [results_eval_emptier[n_time]['objFunction_Fairness'] for n_time in range(n_real_eval)]
    PowerTotal_emptier       = [results_eval_emptier[n_time]['normPower_fair'] for n_time in range(n_real_eval)]
    revFairness_emptier      = [results_eval_emptier[n_time]['normRev_fair'] for n_time in range(n_real_eval)]
    revenueTotal_emptier     = [results_eval_emptier[n_time]['revenueTotal'] for n_time in range(n_real_eval)]
    costPWTotal_emptier      = [results_eval_emptier[n_time]['costPWTotal'] for n_time in range(n_real_eval)]

    Migration_violation_interrupted_jobs_heuristic = [
        results_eval_heuristic[n_time]['migration_violation_interrupted_jobs'] for n_time in range(n_real_eval)]
    Capacity_violation_rejected_new_jobs_heuristic = [
        results_eval_heuristic[n_time]['capacity_violation_rejected_new_jobs'] for n_time in range(n_real_eval)]
    Capacity_violation_mig_interrupted_jobs_heuristic = [
        results_eval_heuristic[n_time]['capacity_violation_mig_interrupted_jobs'] for n_time in range(n_real_eval)]

    capacity_violation_rejected_new_jobs_random    = [results_eval_random[n_time]['rejected_jobs_power_t'] for n_time in
                                                   range(n_real_eval)]
    capacity_violation_rejected_new_jobs_emptier   = [results_eval_emptier[n_time]['rejected_jobs_power_t'] for n_time in
                                                    range(n_real_eval)]
    capacity_violation_rejected_new_jobs_heuristic = [results_eval_heuristic[n_time]['rejected_jobs_power_t'] for n_time
                                                      in range(n_real_eval)]

if not SOLVER_EVALUATION:
    results_solver       = None
    revRatioTotal_solver = None
    PowerTotal_solver    = None
    ObjF_RevCost_solver  = None
    ObjFFairness_solver  = None
else:
    revRatioTotal_solver = [res_solver_ii['RevenueRatioTotal'] for res_solver_ii in results_solver]
    ObjF_RevCost_solver  = [res_solver_ii['Objective_function_rev_minus_cost'] for res_solver_ii in results_solver]
    ObjFFairness_solver  = [res_solver_ii['objFunction_Fairness'] for res_solver_ii in results_solver]
    PowerTotal_solver    = [res_solver_ii['normPower_fair'] for res_solver_ii in results_solver]
    revFairness_solver   = [res_solver_ii['normRev_fair'] for res_solver_ii in results_solver]
    revenueTotal_solver  = [res_solver_ii['revenueTotal'] for res_solver_ii in results_solver]
    costPWTotal_solver   = [res_solver_ii['costPWTotal'] for res_solver_ii in results_solver]


# ----------------------------------------------------------------------------
# PRINTING RESULTS TO CONSOLE
# ----------------------------------------------------------------------------

if POLICY_EVALUATION:
    print('\n +-+-+-+-+-+-+-+ POLICY +-+-+-+-+-+-+-+\n')
    print('ObjF_RevCost', ObjF_RevCost)
    print('revenueTotal', revenueTotal)
    print('costPWTotal', costPWTotal)
    print('penaltyTotal', penaltyTotal)
    print(f'Objective Function Profit Total POLICY         {np.mean(ObjF_RevCost):.4f}')
    print(
        f'Profit:                                        {np.mean(revenueTotal) - np.mean(costPWTotal) - np.mean(penaltyTotal):.2f}')
    print(
        f'Revenue, cost of PW, penalty:                  {np.mean(revenueTotal):.2f}, {np.mean(costPWTotal):.2f}, {np.mean(penaltyTotal):.2f}')
    print(f'Revenue Ratio Total Policy                     {np.mean(revRatioTotal):.4f}')
    print(f'Power Term in Fairness Policy                  {np.mean(PowerTotal):.4f}')
    print(f'Revenue Term in Fairness Policy                  {np.mean(revFairness):.4f}')
    print(f'Objective Function Fairness Total Policy       {np.mean(ObjFFairness):.4f}')
    print()

if BASELINES_EVALUATION:
    print('\n +-+-+-+-+-+-+-+ HEURISTIC +-+-+-+-+-+-+-+\n')
    print('ObjF_RevCost', ObjF_RevCost_heuristic)
    print('revenueTotal', revenueTotal_heuristic)
    print('costPWTotal', costPWTotal_heuristic)
    print('penaltyTotal', penaltyTotal_heuristic)
    print(f'Objective Function Revenue-Cost Total HEURISTIC  {np.mean(ObjF_RevCost_heuristic):.4f}')
    print(
        f'Profit:                                          {np.mean(revenueTotal_heuristic) - np.mean(costPWTotal_heuristic) - np.mean(penaltyTotal_heuristic):.2f}')
    print(
        f'Revenue, cost of PW, penalty:                  {np.mean(revenueTotal_heuristic):.2f}, {np.mean(costPWTotal_heuristic):.2f}, {np.mean(penaltyTotal_heuristic):.2f}')
    print(f'Revenue Ratio Total Heuristic                    {np.mean(revRatioTotal_heuristic):.4f}')
    print(f'Power Term in Fairness Heuristic                 {np.mean(PowerTotal_heuristic):.4f}')
    print(f'Revenue Term in Fairness Heuristic                 {np.mean(revFairness_heuristic):.4f}')
    print(f'Objective Function Fairness Total Heuristic      {np.mean(ObjFFairness_heuristic):.4f}')
    print()

    print('\n +-+-+-+-+-+-+-+ RANDOM +-+-+-+-+-+-+-+\n')

    print('ObjF_RevCost', ObjF_RevCost_random)
    print('revenueTotal', revenueTotal_random)
    print('costPWTotal', costPWTotal_random)
    print(f'Objective Function Revenue-Cost Total RANDOM   {np.mean(ObjF_RevCost_random):.4f}')
    print(
        f'Profit:                                        {np.mean(revenueTotal_random) - np.mean(costPWTotal_random):.2f}')
    print(
        f'Revenue and cost of PW:                        {np.mean(revenueTotal_random):.2f}, {np.mean(costPWTotal_random):.2f}')
    print(f'Revenue Ratio Total Random                     {np.mean(revRatioTotal_random):.4f}')
    print(f'Power Term in Fairness Random                  {np.mean(PowerTotal_random):.4f}')
    print(f'Revenue Term in Fairness Random                {np.mean(revFairness_random):.4f}')
    print(f'Objective Function Fairness Total Random       {np.mean(ObjFFairness_random):.4f}')
    print()

    print('\n +-+-+-+-+-+-+-+ EMPTIER +-+-+-+-+-+-+-+\n')
    print('ObjF_RevCost', ObjF_RevCost_emptier)
    print('revenueTotal', revenueTotal_emptier)
    print('costPWTotal', costPWTotal_emptier)
    print(f'Objective Function Revenue-Cost Total EMPTIER  {np.mean(ObjF_RevCost_emptier):.4f}')
    print(
        f'Profit:                                        {np.mean(revenueTotal_emptier) - np.mean(costPWTotal_emptier):.2f}')
    print(
        f'Revenue and cost of PW:                        {np.mean(revenueTotal_emptier):.2f}, {np.mean(costPWTotal_emptier):.2f}')
    print(f'Revenue Ratio Total Emptier                    {np.mean(revRatioTotal_emptier):.4f}')
    print(f'Power Term in Fairness Emptier                 {np.mean(PowerTotal_emptier):.4f}')
    print(f'Revenue Term in Fairness Emptier               {np.mean(revFairness_emptier):.4f}')
    print(f'Objective Function Fairness Total Emptier      {np.mean(ObjFFairness_emptier):.4f}')
    print()


if SOTA_EVALUATION == 1:
    print('\n +-+-+-+-+-+-+-+ SOTA +-+-+-+-+-+-+-+\n')
    print('ObjF_RevCost', ObjF_RevCost_sota)
    print('revenueTotal', revenueTotal_sota)
    print('costPWTotal', costPWTotal_sota)
    print('penaltyTotal', penaltyTotal_sota)
    print(f'Objective Function Revenue-Cost Total SOTA       {np.mean(ObjF_RevCost_sota):.4f}')
    print(
        f'Profit:                                          {np.mean(revenueTotal_sota) - np.mean(costPWTotal_sota) - np.mean(penaltyTotal_sota):.2f}')
    print(
        f'Revenue, cost of PW, penalty:                  {np.mean(revenueTotal_sota):.2f}, {np.mean(costPWTotal_sota):.2f}, {np.mean(penaltyTotal_sota):.2f}')
    print(f'Revenue Ratio Total SOTA                         {np.mean(revRatioTotal_sota):.4f}')
    print(f'Power Term in Fairness SOTA                      {np.mean(PowerTotal_sota):.4f}')
    print(f'Revenue Term in Fairness SOTA                      {np.mean(revFairness_sota):.4f}')
    print(f'Objective Function Fairness Total SOTA           {np.mean(ObjFFairness_sota):.4f}')
    print()

if CARBONEDGE_EVALUATION == 1:
    print('\n +-+-+-+-+-+-+-+ CARBONEDGE +-+-+-+-+-+-+-+\n')
    print('ObjF_RevCost', ObjF_RevCost_carbonedge)
    print('revenueTotal', revenueTotal_carbonedge)
    print('costPWTotal', costPWTotal_carbonedge)
    print('penaltyTotal', penaltyTotal_carbonedge)
    print('carbonedge_emissions (gCO2)', carbonedge_emissions)
    print(f'Objective Function Revenue-Cost Total CARBONEDGE  {np.mean(ObjF_RevCost_carbonedge):.4f}')
    print(f'Profit:                                           {np.mean(revenueTotal_carbonedge) - np.mean(costPWTotal_carbonedge) - np.mean(penaltyTotal_carbonedge):.2f}')
    print(f'Revenue, cost of PW, penalty:                     {np.mean(revenueTotal_carbonedge):.2f}, {np.mean(costPWTotal_carbonedge):.2f}, {np.mean(penaltyTotal_carbonedge):.2f}')
    print(f'Revenue Ratio Total CarbonEdge                    {np.mean(revRatioTotal_carbonedge):.4f}')
    print(f'Power Term in Fairness CarbonEdge                 {np.mean(PowerTotal_carbonedge):.4f}')
    print(f'Revenue Term in Fairness CarbonEdge               {np.mean(revFairness_carbonedge):.4f}')
    print(f'Objective Function Fairness Total CarbonEdge      {np.mean(ObjFFairness_carbonedge):.4f}')
    print(f'CarbonEdge Total Carbon Emissions (gCO2)          {np.mean(carbonedge_emissions):.2f}')
    print()

if SOLVER_EVALUATION == 1:
    print('\n +-+-+-+-+-+-+-+ SOLVER +-+-+-+-+-+-+-+\n')
    print('ObjF_RevCost', ObjF_RevCost_solver)
    print('revenueTotal', revenueTotal_solver)
    print('costPWTotal', costPWTotal_solver)
    # print('penaltyTotal', penaltyTotal_solver)
    print(f'Objective Function Revenue-Cost Total SOLVER   {np.mean(ObjF_RevCost_solver):.4f}')
    print(
        f'Profit:                                        {np.mean(revenueTotal_solver) - np.mean(costPWTotal_solver):.2f}')
    print(
        f'Revenue and cost of PW:                        {np.mean(revenueTotal_solver):.2f}, {np.mean(costPWTotal_solver):.2f}')
    print(f'Revenue Ratio Total Solver                     {np.mean(revRatioTotal_solver):.4f}')
    print(f'Power Term in Fairness Solver                  {np.mean(PowerTotal_solver):.4f}')
    print(f'Revenue Term in Fairness Solver                {np.mean(revFairness_solver):.4f}')
    print(f'Objective Function Fairness Total Solver       {np.mean(ObjFFairness_solver):.4f}')
    print()


print('\n +-+-+-+-+-+-+-+ OTHER VALUES +-+-+-+-+-+-+-+\n')

# ----------------------------------------------------------------------------
# EXPORT EXTRA METRICS TO TEXT FILE
# Logs violation stats and migration percentages.
# ----------------------------------------------------------------------------

with open(os.path.join(experiment_dir, 'extra_values.txt'), 'w') as fileToSave:
    if POLICY_EVALUATION:
        percent_mig_policy = f'perc_mig             {mean_mig:.2f}'
        print(percent_mig_policy)
        fileToSave.write(percent_mig_policy)
        fileToSave.write('\n')

    if BASELINES_EVALUATION:
        percent_mig_heuris = f'perc_mig heuristic   {mean_migration_heuristic:.2f}'
        print(percent_mig_heuris)
        print()
        fileToSave.write(percent_mig_heuris)
        fileToSave.write('\n')
        fileToSave.write('\n')

    if SOTA_EVALUATION:
        percent_mig_sota = f'perc_mig sota   {mean_migration_sota:.2f}'
        print(percent_mig_sota)
        print()
        fileToSave.write(percent_mig_sota)
        fileToSave.write('\n')
        fileToSave.write('\n')

    if POLICY_EVALUATION:
        violation_mig_policy = f'migration_violation_interrupted_jobs: {np.sum(Migration_violation_interrupted_jobs):.0f}, percentage:  {np.sum(Migration_violation_interrupted_jobs) / (tot_time_evaluation * n_real_eval):.2f}'
        print(violation_mig_policy)
        fileToSave.write(violation_mig_policy)
        fileToSave.write('\n')

        violation_capNew_policy = f'capacity_violation_rejected_new_jobs              {np.sum(Capacity_violation_rejected_new_jobs):.0f},      {np.sum(Capacity_violation_rejected_new_jobs) / (tot_time_evaluation * n_real_eval):.2f}'
        print(violation_capNew_policy)
        fileToSave.write(violation_capNew_policy)
        fileToSave.write('\n')

        violation_capMig_policy = f'capacity_violation_mig_interrupted_jobs           {np.sum(Capacity_violation_mig_interrupted_jobs):.0f},      {np.sum(Capacity_violation_mig_interrupted_jobs) / (tot_time_evaluation * n_real_eval):.2f}'
        print(violation_capMig_policy)
        fileToSave.write(violation_capMig_policy)
        fileToSave.write('\n')
    
    if CARBONEDGE_EVALUATION:
        pass
    # to add if needed

    if BASELINES_EVALUATION:
        violation_mig_heuris = f'migration_violation_heuristic:{np.sum(Migration_violation_interrupted_jobs_heuristic):.0f},         percentage: {np.sum(Migration_violation_interrupted_jobs_heuristic) / (tot_time_evaluation * n_real_eval):.2f}'
        print(violation_mig_heuris)
        print()
        fileToSave.write(violation_mig_heuris)
        fileToSave.write('\n')
        fileToSave.write('\n')

        violation_capNew_heuris = f'capacity_violation_rejected_new_jobs     HEURI    {np.sum(Capacity_violation_rejected_new_jobs_heuristic):.0f},      {np.sum(Capacity_violation_rejected_new_jobs_heuristic) / (tot_time_evaluation * n_real_eval):.2f}'
        print(violation_capNew_heuris)
        fileToSave.write(violation_capNew_heuris)
        fileToSave.write('\n')

        violation_capMig_heuris = f'capacity_violation_mig_interrupted_jobs  HEURI    {np.sum(Capacity_violation_mig_interrupted_jobs_heuristic):.0f},      {np.sum(Capacity_violation_mig_interrupted_jobs_heuristic) / (tot_time_evaluation * n_real_eval):.2f}'
        print(violation_capMig_heuris)
        print()
        fileToSave.write(violation_capMig_heuris)
        fileToSave.write('\n')
        fileToSave.write('\n')

        violation_capNew_random = f'capacity_violation_rejected_new_jobs RANDOM    {np.sum(capacity_violation_rejected_new_jobs_random):.4f},      {np.sum(capacity_violation_rejected_new_jobs_random) / (tot_time_evaluation * n_real_eval):.4f}'
        print(violation_capNew_random)
        fileToSave.write(violation_capNew_random)
        fileToSave.write('\n')

        violation_capNew_emptier = f'capacity_violation_rejected_new_jobs EMPTIER   {np.sum(capacity_violation_rejected_new_jobs_emptier):.4f},      {np.sum(capacity_violation_rejected_new_jobs_emptier) / (tot_time_evaluation * n_real_eval):.4f}'
        print(violation_capNew_emptier)
        print()
        fileToSave.write(violation_capNew_emptier)
        fileToSave.write('\n')
        fileToSave.write('\n')

# ============================================================================
# SUMMARY TABLE (Command Line output)
# ============================================================================
print('\n' + '='*80)
print(' SUMMARY TABLE ')
print('='*80)
print(f'{"Algorithm":<15} {"ObjF":>12} {"Profit":>12} {"Revenue":>12} {"Cost":>12}')
print('-'*80)

if POLICY_EVALUATION:
    profit_policy = np.mean(revenueTotal) - np.mean(costPWTotal) - np.mean(penaltyTotal)
    print(f'{"POLICY":<15} {np.mean(ObjF_RevCost):>12.4f} {profit_policy:>12.2f} {np.mean(revenueTotal):>12.2f} {np.mean(costPWTotal):>12.2f}')

if BASELINES_EVALUATION:
    profit_heuristic = np.mean(revenueTotal_heuristic) - np.mean(costPWTotal_heuristic) - np.mean(penaltyTotal_heuristic)
    print(f'{"HEURISTIC":<15} {np.mean(ObjF_RevCost_heuristic):>12.4f} {profit_heuristic:>12.2f} {np.mean(revenueTotal_heuristic):>12.2f} {np.mean(costPWTotal_heuristic):>12.2f}')
    
    profit_random = np.mean(revenueTotal_random) - np.mean(costPWTotal_random)
    print(f'{"RANDOM":<15} {np.mean(ObjF_RevCost_random):>12.4f} {profit_random:>12.2f} {np.mean(revenueTotal_random):>12.2f} {np.mean(costPWTotal_random):>12.2f}')
    
    profit_emptier = np.mean(revenueTotal_emptier) - np.mean(costPWTotal_emptier)
    print(f'{"EMPTIER":<15} {np.mean(ObjF_RevCost_emptier):>12.4f} {profit_emptier:>12.2f} {np.mean(revenueTotal_emptier):>12.2f} {np.mean(costPWTotal_emptier):>12.2f}')

if SOTA_EVALUATION:
    profit_sota = np.mean(revenueTotal_sota) - np.mean(costPWTotal_sota) - np.mean(penaltyTotal_sota)
    print(f'{"SOTA":<15} {np.mean(ObjF_RevCost_sota):>12.4f} {profit_sota:>12.2f} {np.mean(revenueTotal_sota):>12.2f} {np.mean(costPWTotal_sota):>12.2f}')

if CARBONEDGE_EVALUATION:
    profit_carbonedge = np.mean(revenueTotal_carbonedge) - np.mean(costPWTotal_carbonedge) - np.mean(penaltyTotal_carbonedge)
    print(f'{"CARBONEDGE":<15} {np.mean(ObjF_RevCost_carbonedge):>12.4f} {profit_carbonedge:>12.2f} {np.mean(revenueTotal_carbonedge):>12.2f} {np.mean(costPWTotal_carbonedge):>12.2f}')

if SOLVER_EVALUATION:
    profit_solver = np.mean(revenueTotal_solver) - np.mean(costPWTotal_solver)
    print(f'{"SOLVER":<15} {np.mean(ObjF_RevCost_solver):>12.4f} {profit_solver:>12.2f} {np.mean(revenueTotal_solver):>12.2f} {np.mean(costPWTotal_solver):>12.2f}')

print('='*80 + '\n')
