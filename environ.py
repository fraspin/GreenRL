import os, random, time, copy
import numpy as np
import math as m
from tqdm.auto import tqdm
# import gym
# from gym import spaces
import gymnasium
from gymnasium import spaces

import sys

from sklearn.metrics.pairwise import euclidean_distances
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from energy_datasets import GreenEnergyGenerationDataset


class WorldEnv(gymnasium.Env):
    """
        Custom Environment that follows the Gymnasium interface.
        This environment simulates a Multi-Access Edge Computing (MEC) network
        where an RL agent must allocate or migrate jobs to optimize for profit
        or fairness while managing green/brown energy consumption and delay constraints.
        """

    def reset(self, seed=None, repeat=False, 
              dataTS=None,
              envNoMigration=False, isTraining=True):
        """
                The reset method is called at the beginning of each episode. It returns
                the initial observation and starts the simulation/training process.

                Important: the observation must be a numpy array or a structured dictionary
                containing numpy arrays depending on the defined observation_space.

                :param seed: Random seed for reproducibility.
                :param repeat: If True, uses deterministic data from dataTS instead of random generation.
                :param dataTS: Pre-loaded data for jobs, delays, and green energy (used for evaluation).
                :param envNoMigration: Flag to disable the migration phase (useful for baselines).
                :param isTraining: Flag to indicate if the environment is in training mode.
                :return: current_state_dict (observation), info (empty dict)
                """

        # Load pre-generated data if we are repeating a realization
        jobs             = dataTS.jobs_duration   if dataTS else []
        jobs_wdlay       = dataTS.jobs_wdlay      if dataTS else []
        jobs_arrivalNode = dataTS.jobs_arrivNd    if dataTS else []
        propagationDelay = dataTS.prop_delay      if dataTS else []
        greenpw          = dataTS.greenPW         if dataTS else []


        self.TIME_SLOT        = 0  ## Current TS in the simulation
        self.num_arrived_jobs = 0  ## Total number of arrived jobs

        # Initialize the structured state dictionary
        self.current_state_dict = {"nodes_pc": np.zeros(self.N_SERVERS, dtype=int),
                                   "nodes_green": np.zeros(self.N_SERVERS, dtype=int),
                                   "nodes_migration": np.zeros(self.N_SERVERS, dtype=int),
                                   "index": np.zeros(1, dtype=int)}

        self.next_state_quantized_dict = {"nodes_pc": np.zeros(self.N_SERVERS, dtype=int),
                                          "nodes_green": np.zeros(self.N_SERVERS, dtype=int),
                                          "nodes_migration": np.zeros(self.N_SERVERS, dtype=int),
                                          "index": np.zeros(1, dtype=int)}

        # Raw state array used internally before quantization
        self.next_state_raw = np.zeros(self.N_SERVERS + self.N_SERVERS + self.N_SERVERS + 1, dtype=int)

        # Buffers for calculating metrics over a sliding window
        self.reward     = 0  ## reward of the current action
        self.old_reward = 0  ## reward of the past action
        self.ts_repeat  = 0  ## TS if we are repeating a realization for test

        self.bufferAcceptance = np.zeros((self.T_buffer, 2))  # 1st dimension accepted, 2nd arrived
        self.bufferRevenue    = np.zeros((self.T_buffer, 2))  # 1st dimension accepted, 2nd arrived
        self.bufferPower      = np.zeros((self.T_buffer, 1))

        self.n_steps = 0

        # Evaluation metrics for tracking constraint violations
        self.error_migration             = 0  # for evaluation ##############################
        self.lastJobExceededNodePw       = 0  # for evaluation ##############################
        self.lastJobExceededDeadline     = 0  # for evaluation ##############################
        self.newJobExceededNodePw        = 0  # for evaluation ##############################
        self.migratedJobExceededNodePw   = 0  # for evaluation ##############################
        self.newJobExceededDeadline      = 0  # for evaluation ##############################
        self.migratedJobExceededDeadline = 0  # for evaluation ##############################
        self.error_capacity              = 0  # for evaluation ##############################

        # Configuration flags
        self.envWithNoMigration = envNoMigration
        self.envIsTraining = isTraining

        if envNoMigration and isTraining:
            sys.err("Impossible to be at the same time training and evaluating a baseline")

        # System state trackers for jobs located at each node
        self.Node_List         = [[]] * self.N_SERVERS
        self.startTS_Node_list = [[]] * self.N_SERVERS
        self.DuratList         = [[]] * self.N_SERVERS
        self.startTS_DuratList = [[]] * self.N_SERVERS
        self.DelayList         = [[]] * self.N_SERVERS
        self.startTS_DelayList = [[]] * self.N_SERVERS

        self.jobs_migratedList = [[]] * self.N_SERVERS  ## maybe check

        # Number of migrations initiated at each server in the current TS
        self.migration = np.zeros(self.N_SERVERS)

        ## Index of the current job
        self.jobIndex = 0

        ## Binary variable to set if we are starting a new TS
        self.isNewTS = 1

        ### Saving if we are repeating the environment (baselines) or not
        self.repeatRealization = repeat

        """
        Energy Generation:
        Generate or retrieve green energy values depending on if we are training or testing.
        """
        if self.repeatRealization == False:
            if isTraining: #if we are training, we generate random green energy values
                self.next_state_raw[self.N_SERVERS: 2 * self.N_SERVERS] = self.greenEnergyGeneration()
            else: # if we are not training, we load pre-generated green energy values from datasets
                self.next_state_raw[self.N_SERVERS: 2 * self.N_SERVERS] = GreenEnergyGenerationDataset(self.N_SERVERS, self.MAX_PW_SERVERS, 0)
        else:
            self.next_state_raw[self.N_SERVERS: 2 * self.N_SERVERS] = greenpw[:, 0]

        self.green_power_levels = self.next_state_raw[self.N_SERVERS: 2 * self.N_SERVERS]

        """
        Network/Node Geometry and Delays Setup
        """
        if self.repeatRealization:
            self.jobsRealization = jobs
            self.wlessDelayRealization = jobs_wdlay
            self.arrivNodeRealization = jobs_arrivalNode
            self.matrix_propagation_delay = propagationDelay[:, :]
        else:  # if self.repeatRealization == False:
            # Generate random locations for servers and compute propagation delays
            self.serverLocations = np.random.uniform(size=[self.N_SERVERS, 2]) * self.SIZE_KM_SETTING_SIDE
            self.serversDistances = euclidean_distances(self.serverLocations, self.serverLocations)
            self.matrix_propagation_delay = self.serversDistances * 5e-3  # speed og light fiber = 200 m/micro-sec -> 5 microsec/km -> 0.005 ms/km

        # Generate the first batch of jobs for the starting TS
        self.jobs_list_timeS, self.jobs_list_wdlay, self.job_list_arrivNode = self.generateNewJobs()

        # Initialize the delay vector for the first job (if any)
        if  self.jobs_list_timeS != []:
            self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = self.getTotalDelay(isaNewJob=True)

        # Quantize the raw state to fit into the MultiDiscrete observation space
        self.next_state_quantized_dict = self.state_quantization(self.next_state_raw)
        self.current_state_dict = self.next_state_quantized_dict

        return self.current_state_dict, {}

    def __init__(self,
                 num_servers,  ## Number of edge servers/nodes
                 max_pc_server,  ## Normalized processing cycles per node per time slot
                 arrivalRate,  ## Rate of arrived jobs per TS
                 max_pw_server,  ## Normalized power capacity per node per time slot
                 jobSessionLength,  ## Job duration in  TS
                 pw_params,  ## Parameters to translate processing to power
                 obj_params,  ## Parameters for objective function
                 revenue_factor,  ## Normalized factor of revenue per job
                 cost_factor,  ## Normalized factor of cost per unit of power
                 c_jn,  ## Normalized processing cycles per job per time slot
                 computationDelay,  ## Delay (in ms) of computation in the server 
                 switchingDelay,  ## Delay (in ms) of swithcing in backhaul
                 deadlineDelay,  ## Delay (in ms) maximum for QoE
                 minWirelessRate,  ## Minimum wireless data rate for any user
                 maxWirelessRate,  ## Maximum wireless data rate for any user
                 fiberDataRate,  ## Data rate for network links
                 frameSize,  ## Size of video frame in XXXXX
                 maxStorageServer,  ## Maximum storage capacity of server 
                 job_storage,  ## Required storage for each job
                 T_buffer=10,  ## Window sizez to evaluate reward during training
                 window_m=1,  ## duration of migration (in TS)
                 revCost_function=1,  ## 1 if (profit P1), 0 if fairness (P2)
                 nolog_fairness=0,  ## 1 = fairness objective without log. 0 if log
                 setting_size=50  ## Size of the square where edge nodes are located
                 ):
        """
        INIT method
        """
        super(WorldEnv, self).__init__()

        self.epsilon_reward = 1
        self.N_SERVERS = num_servers  ## Number of edge servers/nodes
        self.fiber_datarate = fiberDataRate
        self.min_datarate = minWirelessRate
        self.max_datarate = maxWirelessRate
        self.FRAME_SIZE = frameSize
        self.COMPUTATION_DELAY = computationDelay
        self.SWITCHING_DELAY = switchingDelay
        self.DEADLINE_DELAY = deadlineDelay
        # Calculate standard transmission delay across fiber
        self.TRASMISSION_DELAY = (self.FRAME_SIZE / self.fiber_datarate) * 1000 + self.SWITCHING_DELAY  # in ms
        self.COST_FACTOR = cost_factor
        self.REV_FACTOR = revenue_factor  # proportial coefficient for revenues
        self.arrivalRate = arrivalRate
        self.jobSessionLength = jobSessionLength  # Number of jobs in the system - useful for defining action space and observation space
        self.JOB_STORAGE = job_storage

        # Normalize server capacity arrays
        if not isinstance(max_pc_server, list):
            self.MAX_PC_SERVERS = [max_pc_server] * self.N_SERVERS
        else:
            self.MAX_PC_SERVERS = max_pc_server

            ## maximum processing cycles for server
        if not isinstance(max_pw_server, list):
            self.MAX_PW_SERVERS = [max_pw_server] * self.N_SERVERS
        else:
            self.MAX_PW_SERVERS = max_pw_server

            ## maximum processing cycles for server
        if not isinstance(maxStorageServer, list):
            self.MAX_STORAGE_SERVER = [maxStorageServer] * self.N_SERVERS
        else:
            self.MAX_STORAGE_SERVER = maxStorageServer


        self.SIZE_KM_SETTING_SIDE = setting_size
        self.jobsRealization = []
        self.jobsRealizationDelay = []
        self.c_jn = c_jn

        # Power and Objective weights
        self.alpha, self.beta, self.gamma = pw_params  # tuple of 3 values of proc-cycles to power conversor
        self.rho_d, self.rho_p, self.rho_r = obj_params  # factors to weight delay, power, and revenue

        self.T_buffer = T_buffer
        self.window_m = window_m

        self.revcost_function = revCost_function  # 1 if profit, 0 if fairness
        self.noLog_fairness = nolog_fairness  # 1 if no log in fairness, 0 if log

        #### Initializing quantization configuration for the STATE space
        min_server = np.argmin(self.MAX_PC_SERVERS)
        self.server_num_steps_green = [3] * self.N_SERVERS  ## for quantizing the state
        self.quantPoints, self.n_quant_step = self.getQuantizationPoints(c_jn, self.MAX_PC_SERVERS[
            min_server])  # n_quant_step
        self.server_num_steps_pc = [self.n_quant_step] * self.N_SERVERS  # [x+1 for x in self.max_pc_server2] # n_quant_step

        # ----------------- DEFINE ACTION SPACE -----------------
        # Action space spans 0 to N_SERVERS.
        # 0 = Reject job (if new) / Interrupt job (if migrating)
        # 1 to N = Node ID to allocate/migrate the job to.
        self.action_space = spaces.Discrete(self.N_SERVERS + 1)

        # ----------------- DEFINE OBSERVATION SPACE -----------------
        # The observation space is a Dictionary holding quantized features.
        self.binary_state_migration = [2] * self.N_SERVERS
        self.observation_space = spaces.Dict({
            "nodes_pc": spaces.MultiDiscrete(np.array(self.server_num_steps_pc, dtype=np.int32)),
            "nodes_green": spaces.MultiDiscrete(np.array(self.server_num_steps_green, dtype=np.int32)),
            "nodes_migration": spaces.MultiDiscrete(np.array(self.binary_state_migration, dtype=np.int32)),
            "index": spaces.Discrete(self.N_SERVERS+1)
        })
        # Call reset to safely initialize internal variables
        self.reset()

    def step(self, action):
        """
                Main environment step. Processes the action chosen by the agent.
                The step function is called sequentially for EACH job in the system.
                Phase 1: Iterate over existing jobs to decide on migrations.
                Phase 2: Iterate over newly arrived jobs to decide on allocations.
                When the last job is processed, it transitions the Time Slot (TS).

                :param action: Chosen integer from action_space.
                :return: next_state, reward, done, truncated, info
                """

        # Sync the start of TS status
        if self.isNewTS:
            self.Node_List = copy.deepcopy(self.startTS_Node_list)
            self.DuratList = copy.deepcopy(self.startTS_DuratList)
            self.DelayList = copy.deepcopy(self.startTS_DelayList)
            # self.arrNdList  = copy.deepcopy(self.startTS_arrNdList) 
            self.isNewTS = 0  ## Next step is not going to be in new TS

        self.n_steps += 1  # One more step beyond...

        done = False  ## Default: Everything works.
        isTSConcluded = False  ## Flag to now when all jobs are served in this TS


        self.next_state_quantized_dict = self.current_state_dict
        self.green_power_levels = self.next_state_raw[self.N_SERVERS: 2 * self.N_SERVERS]

        # Get the node where the current job is located (0 if it is a newly arrived job)
        nodeofJob = self.current_state_dict['index']  #################################### checkkkkkkkk!!!!!

        # Binary reachability vector (1 = unreachable due to delay constraint)
        available_nodes = self.current_state_dict['nodes_migration']

        info = {}
        if nodeofJob != 0:  ## nodeofJob == 0 implies job is newly arrived.
            ## In this case, action cannot be 0 because the job must not be interrupted.
            # ================= MIGRATION PHASE =================
            self.lastJobWasInterrupted = 0  ## Initiating the variable as OK (not used)

            if action != 0:  ## Ok, not interrupted by policy
                if available_nodes[action - 1] != 1:  # Node is reachable with low latency
                    done = self.handleJobMigration(action, nodeofJob)
                else:  # Node is NOT reachable with low latency
                    """ The node is not available due to migration delay constraint 
                    but nevertheless the algorithm  allocates the job to it
                    """
                    done = self.handleErrorInMigration(nodeofJob)

            else:  ## action == 0 with nodeOfJob != 0 -> allocated job is interrupted, error.
                done = self.handleErrorInMigration(nodeofJob)

            ### saving actions job index as info
            info = {'jobIndex': self.jobIndex}

            """ Get index of next job to evaluate migration. If finished, move to 
                allocate new jobs. """
            isTSConcluded = self.migrateNextJoborPassToNew(nodeofJob)

        else:  ## nodeofJob == 0 -> new job
            # ================= NEW ALLOCATION PHASE =================
            if len(self.jobs_list_timeS):
                done, isTSConcluded = self.handleNewJob(action)
            else:
                isTSConcluded = True

        # If all jobs are processed, move simulation time forward
        if isTSConcluded:
            self.handleChangeTS()

        self.next_state_quantized_dict = self.state_quantization(self.next_state_raw)
        self.current_state_dict        = self.next_state_quantized_dict

        truncated = False

        return self.current_state_dict, self.reward, done, truncated, info

    # =========================================================================
    # LOGIC HANDLING FUNCTIONS
    # =========================================================================

    def getTotalDelay(self, isaNewJob, nodeofJob=None, jobIndexInNode=None):
        """
        Determines the reachability of nodes for a given job by calculating the total delay.
        Total Delay = (Wireless Delay + Propagation Delay + Transmission Delay) * 2 + Computation Delay.

        :return: A binary numpy array where 1 indicates a node is unreachable.
        """

        if isaNewJob == 1:
            ## We randomly choose the node (or BS) where the job arrives.
            ## And we obtain the corresponding wireless delay
            nodeofJob  = self.job_list_arrivNode[0]
            job_wDelay = self.jobs_list_wdlay[0]

        else:  ## we obtain the corresponding wireless delay
            job_wDelay = self.DelayList[nodeofJob - 1][jobIndexInNode]

        ## Obtaining the corresponding propagation delay towards all nodes
        propDelayNodeofJob = self.matrix_propagation_delay[nodeofJob - 1, :]

        ## Initialize binary vector
        nodeTooFar = np.zeros(self.N_SERVERS)  # nodeTooFar[n] = 1 if delay is too long

        for n in range(self.N_SERVERS):
            if (job_wDelay + propDelayNodeofJob[n] + self.TRASMISSION_DELAY) * 2 \
                    + self.COMPUTATION_DELAY > self.DEADLINE_DELAY:  #\
                nodeTooFar[n] = 1
        return nodeTooFar

    def getNextNodeWithJobs(self, node_idx):
        """Finds the next server node that contains active jobs."""
        if node_idx < self.N_SERVERS:
            while node_idx < self.N_SERVERS and len(self.startTS_Node_list[node_idx]) == 0:
                node_idx += 1
        else:  # node_idx == self.N_SERVERS:
            node_idx = self.N_SERVERS

        return node_idx + 1

    def handleJobMigration(self, action, nodeofJob):
        """Executes the migration of a job from nodeofJob to the target action node."""
        job_pc           = self.startTS_Node_list[nodeofJob - 1][self.jobIndex]
        job_remaining_TS = self.startTS_DuratList[nodeofJob - 1][self.jobIndex]
        delay_job        = self.startTS_DelayList[nodeofJob - 1][self.jobIndex]

        if action != nodeofJob:
            self.Node_List[nodeofJob - 1][self.jobIndex] = 0
            self.DuratList[nodeofJob - 1][self.jobIndex] = 0
            self.DelayList[nodeofJob - 1][self.jobIndex] = 0

            self.migration[action - 1] += int(1)

            if self.Node_List[action - 1] == []:
                self.Node_List[action - 1] = [job_pc]
                self.DuratList[action - 1] = [job_remaining_TS]  # remaining duration of job
                self.DelayList[action - 1] = [delay_job]
                self.jobs_migratedList[nodeofJob - 1] = [1]
            else:
                self.Node_List[action - 1].append(job_pc)
                self.DuratList[action - 1].append(job_remaining_TS)
                self.DelayList[action - 1].append(delay_job)
                self.jobs_migratedList[nodeofJob - 1].append(1)
        else:  ## if action == nodeofJob, do nothing
            pass

        ### Compute next state for processing cycles 
        self.updatePcState()

        """ We check if the job migrated fulfill the capacity constraint and we assign a reward or penalty accordingly"""
        done = self.objectiveVerification(action, isNewJob=False)  # if true reward = penalty

        return done

    def handleErrorInMigration(self, nodeofJob):
        """Handles an invalid migration attempt (triggers a terminal state penalty)."""
        done = True  # Error -> stop training iteration

        ### Modify lists (they are used for updating the state)
        self.Node_List[nodeofJob - 1][self.jobIndex] = 0
        self.DuratList[nodeofJob - 1][self.jobIndex] = 0
        self.DelayList[nodeofJob - 1][self.jobIndex] = 0

        ### Compute next state for processing cycles
        self.updatePcState()

        ### Accounting for the error
        self.error_migration += 1  ## One more error trying to migrate
        self.lastJobWasInterrupted = 1  ## Eventually, we interrupted the job

        ### Assign reward due to error: 
        self.reward = -1

        return done

    def migrateNextJoborPassToNew(self, nodeofJob):
        """Advances the iteration to the next job in the migration phase, or to the allocation phase."""
        endOfTS = False  # Flag to know if all jobs have been served and we move to next TS

        ### pass to next job and update state accordingly
        if self.jobIndex + 1 >= len(self.startTS_Node_list[nodeofJob - 1]):
            ## if there are NOT any more jobs to handle in this node, jump to new node
            self.jobIndex = 0
            nextNode = self.getNextNodeWithJobs(nodeofJob)
        else:  ## if there are still jobs to handle in this node:
            self.jobIndex = self.jobIndex + 1
            nextNode = nodeofJob

        """ if we move to a new node with new jobs, we need to change also 
        the reachability of nodes because there is both a new distance between nodes and a new job latency
        """
        if nextNode <= self.N_SERVERS:  ## still jobs in other servers
            self.next_state_raw[3 * self.N_SERVERS] = nextNode
            isaNewJob = False  # we are still migrating
            NewMigrationList = self.getTotalDelay(isaNewJob, nextNode, self.jobIndex)
            # print('NewMigrationList', NewMigrationList)
            self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = NewMigrationList

        else:  ## nextNode > self.N_SERVERS: all jobs in system have been checked, pass to newly arrived jobs
            self.next_state_raw[3 * self.N_SERVERS] = 0  # New job not in any node yet
            self.jobIndex = 0


            if len(self.jobs_list_timeS) == 0 and self.next_state_raw[3 * self.N_SERVERS] == 0:
                endOfTS = True
            else:
                isaNewJob = True
                NewMigrationList = self.getTotalDelay(isaNewJob)
                # print('NewMigrationList1', NewMigrationList)
                self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = NewMigrationList

        return endOfTS

    # %%% Function to handle the acceptance of an incoming job

    def handleNewJob(self, action):
        """Handles the allocation of a newly arrived job or logs its rejection."""
        isAccepted = 0 if action == 0 else 1

        self.bufferAcceptance[0][0] += isAccepted
        self.bufferAcceptance[0][1] += 1
        self.bufferRevenue[0][0] += isAccepted * self.REV_FACTOR * self.c_jn * self.jobs_list_timeS[
            0]  ## change if pc job change
        self.bufferRevenue[0][1] += self.REV_FACTOR * self.c_jn * self.jobs_list_timeS[0]  ## hange if pc job change
        self.num_arrived_jobs += 1

        ##### ALLOCATION JOBS #####
        if action != 0:  # job accepted
            selectedNode = action - 1

            if self.Node_List[selectedNode] == []:
                self.Node_List[selectedNode] = [int(self.c_jn)]
                self.DuratList[selectedNode] = [int(self.jobs_list_timeS[0])]  ### D_MAX = Duration of job
                self.DelayList[selectedNode] = [int(self.jobs_list_wdlay[0])]

            else:
                self.Node_List[selectedNode].append(int(self.c_jn))
                self.DuratList[selectedNode].append(int(self.jobs_list_timeS[0]))
                self.DelayList[selectedNode].append(int(self.jobs_list_wdlay[0]))
        else:  # job rejected
            pass

        del self.jobs_list_timeS[0]
        del self.jobs_list_wdlay[0]
        del self.job_list_arrivNode[0]

        ### Compute next state (processing cycles)
        self.updatePcState()
        self.next_state_raw[3 * self.N_SERVERS] = 0  ## indicating that the following job is new
        ## if not, it will be updated later in handleEndTS

        done = self.objectiveVerification(action, isNewJob=True)  # if true reward = penalty
        endOfTS = True if len(self.jobs_list_timeS) == 0 else False

        if len(self.jobs_list_timeS) != 0:
            isaNewJob = True
            NewMigrationList = self.getTotalDelay(isaNewJob)
            self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = NewMigrationList

        return done, endOfTS

    def removeEmptyJobSlots(self, variable_jobsInNodes):
        """Cleans up internal lists by removing terminated or migrated jobs (zeros)."""
        return [[job for job in node if job != 0] for node in variable_jobsInNodes]

    def handleChangeTS(self):
        """Advances the Time Slot, updating buffers, decreasing job durations, and spawning new jobs."""

        ### Update the lists to remove empty spots left by migrated jobs
        self.Node_List = self.removeEmptyJobSlots(self.Node_List)
        self.DuratList = self.removeEmptyJobSlots(self.DuratList)
        self.DelayList = self.removeEmptyJobSlots(self.DelayList)

        ### Compute next state (processing cycles)
        self.updatePcState()

        # saving current node list for next TS
        self.startTS_Node_list = copy.deepcopy(self.Node_List)
        self.startTS_DuratList = copy.deepcopy(self.DuratList)
        self.startTS_DelayList = copy.deepcopy(self.DelayList)
        self.jobs_migratedList = [[]] * self.N_SERVERS

        ### Decrease duration of jobs and update power/length if job is finished
        for node_idx, jobsInNode in enumerate(self.startTS_DuratList):
            for job_idx, jobLength in enumerate(jobsInNode):  # for each job in the node
                self.startTS_DuratList[node_idx][job_idx] = jobLength - 1
                if jobLength - 1 <= 0:  # get lists ready to remove values
                    self.startTS_Node_list[node_idx][job_idx] = 0
                    self.startTS_DuratList[node_idx][job_idx] = 0
                    self.startTS_DelayList[node_idx][job_idx] = 0

        ### Remove jobs that have been finished
        self.startTS_Node_list = self.removeEmptyJobSlots(self.startTS_Node_list)
        self.startTS_DuratList = self.removeEmptyJobSlots(self.startTS_DuratList)
        self.startTS_DelayList = self.removeEmptyJobSlots(self.startTS_DelayList)

        self.isNewTS    = 1
        self.ts_repeat += 1
        self.jobIndex   = 0


        ##### ------ UPDATE OF BUFFERS ---------- #######
        self.bufferAcceptance = np.roll(self.bufferAcceptance, 1, axis=0)
        self.bufferRevenue    = np.roll(self.bufferRevenue, 1, axis=0)
        self.bufferPower      = np.roll(self.bufferPower, 1)

        self.bufferAcceptance[0, :] = 0
        self.bufferRevenue[0, :]    = 0
        self.bufferPower[0, :]      = 0

        ##### ---- UPDATE TIME SLOT AND CHECK END OF EPISODE ---- ######
        self.migration = np.zeros(self.N_SERVERS)
        self.TIME_SLOT += 1

        self.error_migration           = 0
        self.newJobExceededNodePw      = 0  # for evaluation ###########
        self.migratedJobExceededNodePw = 0  # for evaluation ###########
        self.error_capacity            = 0

        self.jobs_list_timeS, self.jobs_list_wdlay, self.job_list_arrivNode = self.generateNewJobs()  ### change so that it returns a vector of size = num jobs and each element = duration of job        

        if self.envIsTraining:  ## training phase
            ## If we dont have jobs in system, we do not perfom migrations
            jumpToMigrations = any(self.startTS_Node_list)
            ## if there are no jobs in system and no job arrives next TS:
            noJobsForNextTS = (not any(self.startTS_Node_list)) \
                              and (not any(self.jobs_list_timeS))


            ## if noJobsForNextTS, we advance one T
            if noJobsForNextTS:
                self.handleChangeTS()
            else:
                if jumpToMigrations:
                    """ If there are jobs in the system, we focus on migration """
                    nextNodeWithJobs = self.getNextNodeWithJobs(0)
                    self.next_state_raw[3 * self.N_SERVERS] = nextNodeWithJobs
                    self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = self.getTotalDelay(isaNewJob=False,
                                                                                                     nodeofJob=nextNodeWithJobs,
                                                                                                     jobIndexInNode=0)
                else:
                    """ If there are NOT jobs in the system, we move to acceptance"""
                    self.next_state_raw[3 * self.N_SERVERS] = 0
                    self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = self.getTotalDelay(isaNewJob=True)


        else:  ## algorithm evaluation

            if self.envWithNoMigration:
                ## When evaluating baselines, we do not perfom migrations
                jumpToMigrations = False
                ## if no job arrives next TS:
                noJobsForNextTS = (not any(self.jobs_list_timeS))
            else:
                ## If we dont have jobs in system, we do not perfom migrations
                jumpToMigrations = any(self.startTS_Node_list)

                ## if there are no jobs in system and no job arrives next TS:
                noJobsForNextTS = (not any(self.startTS_Node_list)) \
                                  and (not any(self.jobs_list_timeS))

                ## if noJobsForNextTS, we advance one T
            if noJobsForNextTS:
                # print(f"ts_repeat={self.ts_repeat} - NO JOBS TO HANDLE, PASSING TO NEW TS AGAIN")
                self.next_state_raw[3 * self.N_SERVERS] = 0
                self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = np.zeros(self.N_SERVERS)
            else:
                if jumpToMigrations:
                    """ If there are jobs in the system, we focus on migration """
                    nextNodeWithJobs = self.getNextNodeWithJobs(0)
                    self.next_state_raw[3 * self.N_SERVERS] = nextNodeWithJobs
                    self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = self.getTotalDelay(isaNewJob=False,
                                                                                                     nodeofJob=nextNodeWithJobs,
                                                                                                     jobIndexInNode=0)

                else:
                    """ If there are NOT jobs in the system but there are new ones, we move to acceptance"""
                    self.next_state_raw[3 * self.N_SERVERS] = 0
                    self.next_state_raw[2 * self.N_SERVERS: 3 * self.N_SERVERS] = np.zeros(self.N_SERVERS)

    # =========================================================================
    # OBJECTIVE / REWARD VERIFICATION
    # =========================================================================
    def objectiveVerification(self, action, isNewJob):
        """
        Validates constraints (Power/Capacity and Delay) and calculates the reward.
        Returns a done flag (True if an episode-ending violation occurred).
        """
        done = False  # initializing it to OK

        ##### --------- POWER VERIFICATION ----------- #########################
        ### check if maximum capacity of nodes is exceeded
        self.lastJobExceededNodePw = 0  # for evaluation
        self.lastJobExceededDeadline = 0  # for evaluation
        for nn in range(self.N_SERVERS):
            if self.next_state_raw[nn] > self.MAX_PC_SERVERS[nn]:
                # self.reward = - self.n_steps
                self.reward = -1
                self.error_capacity += 1

                if isNewJob == 1:
                    self.newJobExceededNodePw += 1  # for evaluation
                    # print('VIOLATED FOR EXCEEDING ALLOCATION')
                    # print('due to New job allocated')
                else:
                    self.migratedJobExceededNodePw += 1  # for evaluation
                    # print('VIOLATED FOR EXCEEDING MIGRATION')
                    # print('due to job migrated')
                self.lastJobExceededNodePw = 1  # for evaluation


                self.Node_List[action - 1] = self.Node_List[action - 1][:-1]
                self.DuratList[action - 1] = self.DuratList[action - 1][:-1]
                self.DelayList[action - 1] = self.DelayList[action - 1][:-1]

                # print("\n%%%%%%%%%%%%%%%%%%%%%%%%% ---- ",
                #       "NODES CAPACITY EXTENDED ----- ",
                #       "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                self.updatePcState()

                done = True

        if self.next_state_raw[2 * self.N_SERVERS + action - 1] == 1 and done == False:

            self.reward = -1

            if isNewJob == 1:
                self.newJobExceededDeadline += 1  # for evaluation

            else:
                self.migratedJobExceededDeadline += 1  # for evaluation

            self.lastJobExceededDeadline = 1  # for evaluation


            self.Node_List[action - 1] = self.Node_List[action - 1][:-1]
            self.DuratList[action - 1] = self.DuratList[action - 1][:-1]
            self.DelayList[action - 1] = self.DelayList[action - 1][:-1]

            self.updatePcState()

            done = True

        ### computation of power consumed per job per node
        nodePW = self.getPowerConsumption()

        ### computation of brown energy consumed
        total_no_renewable_power = 0
        for nn in range(self.N_SERVERS):
            no_renewable_power = max(nodePW[nn] - self.next_state_raw[self.N_SERVERS + nn], 0)
            total_no_renewable_power += no_renewable_power

        self.bufferPower[0] = total_no_renewable_power
        brownPower_maximum = np.sum(self.MAX_PW_SERVERS) - np.sum(self.green_power_levels)
        if brownPower_maximum == 0:
            power_window = 0
        else:
            power_window = np.sum(self.bufferPower) / (brownPower_maximum * self.T_buffer)

        cost_brown = np.sum(self.bufferPower) * self.COST_FACTOR

        obj_norm_power = 1 - self.rho_p * power_window

        ##### --------- REVENUE VERIFICATION ----------- #######################
        sumAccepted, sumArrived = np.sum(self.bufferAcceptance, axis=0)
        if sumArrived != 0:
            obj_acceptRatio = sumAccepted / sumArrived
        else:
            obj_acceptRatio = 0
        sumRevAccepted, sumRevArrived = np.sum(self.bufferRevenue, axis=0)

        ############### METRICS ################################################

        ###### IF PROFIT OBJ ######
        obj_revenue = sumRevAccepted

        if self.revcost_function:
            if sumRevArrived != 0:
                possible_reward = (sumRevAccepted - cost_brown) / sumRevArrived
            else:
                possible_reward = 1
        else:
            if sumRevArrived != 0:
                obj_norm_revenue = (sumRevAccepted / sumRevArrived) * self.rho_r + (1 - self.rho_r)
            else:
                obj_norm_revenue = 1
            # reward_nolog    = sumRevAccepted * obj_norm_power # * obj_norm_dela
            possible_reward = obj_norm_revenue * obj_norm_power
            if not self.noLog_fairness:
                possible_reward = m.log(possible_reward + self.epsilon_reward)
                possible_reward = possible_reward / m.log(1 + self.epsilon_reward)

        ########################################################################


        if done == False:
            self.reward = possible_reward

        self.old_reward = possible_reward

        return done

    # =========================================================================
    # STATE AND METRICS HELPERS
    # =========================================================================
    def updatePcState(self):
        """Updates the raw state vector array with the current sum of processing cycles."""
        self.next_state_raw[:self.N_SERVERS] = [int(np.sum(jobs_pc_list)) for jobs_pc_list in self.Node_List]

    def getPowerConsumption(self):
        """Calculates total power consumed by each server based on active jobs and migrations."""
        power = np.zeros(self.N_SERVERS)
        for nodeIdx, nodeJobs in enumerate(self.Node_List):
            for jobPC in nodeJobs:
                power[nodeIdx] += (self.alpha * jobPC + self.gamma)

            power[nodeIdx] += self.beta * self.migration[nodeIdx]

        return power

    ############################################################################
    #%% Functions for quantization of the state
    ############################################################################

    def state_quantization(self, raw_state):
        """Translates the continuous/unbounded numerical state arrays into a discrete dictionary structure."""
        quantized_state = np.zeros(3 * self.N_SERVERS + 1, dtype=int)
        quantized_state_dict = {"nodes_pc": np.zeros(self.N_SERVERS, dtype=int),
                                "nodes_green": np.zeros(self.N_SERVERS, dtype=int),
                                "nodes_migration": np.zeros(self.N_SERVERS, dtype=int),
                                "index": np.zeros(0, dtype=int)}

        quantized_state[:self.N_SERVERS] = [self.pc_quantization(val / self.MAX_PC_SERVERS[index]) for
                                            index, val in enumerate(raw_state[:self.N_SERVERS])]
        quantized_state_dict["nodes_pc"] = quantized_state[:self.N_SERVERS]

        nodePW = self.getPowerConsumption()

        for ii in range(self.N_SERVERS):
            quantized_state[self.N_SERVERS + ii] = self.greenPW_quantization(
                nodePW[ii] - raw_state[self.N_SERVERS + ii])
            quantized_state[2 * self.N_SERVERS + ii] = raw_state[2 * self.N_SERVERS + ii]

        quantized_state_dict["nodes_green"] = quantized_state[self.N_SERVERS: 2 * self.N_SERVERS]
        quantized_state_dict["nodes_migration"] = raw_state[2 * self.N_SERVERS: 3 * self.N_SERVERS]

        quantized_state[3 * self.N_SERVERS] = int(raw_state[3 * self.N_SERVERS])
        quantized_state_dict["index"] = int(quantized_state[3 * self.N_SERVERS])

        return quantized_state_dict

    def greenPW_quantization(self, raw_value):
        """Quantizes green energy bounds (0: enough green, 1: starting brown, 2: highly brown)."""
        if raw_value <= - self.c_jn:
            next_state_quantized = 0  # green energy state = enough green energy
        elif raw_value < 0:
            next_state_quantized = 1  # green energy state = starting to use brown energy
        else:
            next_state_quantized = 2  # green energy state = already using brown energy

        return next_state_quantized

    def pc_quantization(self, raw_value):
        """Maps continuous processing cycle percentage onto a discrete scale."""
        return np.argmax(raw_value < self.quantPoints) - 1

    def getQuantizationPoints(self, pw_job, pw_max):  # n_steps
        """Creates the logarithmic boundaries for quantizing processing cycle states."""
        pw_job_n = pw_job / pw_max

        # but some times there would be space for more jobs.
        min_n_enough = 9 * (10 ** (pw_job_n - 1)) / (10 ** pw_job_n - 1)
        n_steps = int(np.ceil(min_n_enough))  # just getting the closest integer that fits

        base_granularity = 1000
        axis = np.linspace(1, 10, int(n_steps * base_granularity))
        log_val = np.log(axis)
        log_norm_val = log_val / np.max(log_val)

        qvector = [log_norm_val[int(ii * base_granularity)] for ii in range(n_steps)] + [1.01]

        return np.array(qvector), n_steps

    # =========================================================================
    # RANDOM EVENT GENERATION
    # =========================================================================

    def greenEnergyGeneration(self, current_energy=[], node=None):

        if node == None:  ## return full vector
            new_energy = [round(random.randint(2, self.MAX_PW_SERVERS[node])) for node in range(self.N_SERVERS)]
            #the following is for the low green energy scenario
            # new_energy = [round(random.randint(2, 4)) for node in range(self.N_SERVERS)]
        else:  ## return just one server
            new_energy = round(random.randint(2, self.MAX_PW_SERVERS[node]))
        return new_energy

    def greenEnergyGeneration1(self, current_energy=[], node=None):
        if node == None:  ## return full vector
            new_energy = [round(random.randint(2, self.MAX_PW_SERVERS[node])) for node in range(self.N_SERVERS)]
        else:  ## return just one server
            new_energy = round(random.randint(2, self.MAX_PW_SERVERS[node]))
        return new_energy

    def newJobSize(self, max_cycles):
        ''' newJobSize returns the length in TS of the job. To be extended if needed
        to be a random length within some range of TSs '''

        return self.jobSessionLength

    def generateNewJobs(self):
        """
        Generates incoming jobs using a Poisson process for arrivals,
        assigning random wireless delays and arrival nodes.
        """

        if self.repeatRealization:  # in case we are not generating randomly the jobs but getting them from previous experiments
            if self.ts_repeat < len(self.jobsRealization):
                jobList       = [job for job in self.jobsRealization[self.ts_repeat]]
                jobList_wdlay = [job_wd for job_wd in self.wlessDelayRealization[self.ts_repeat]]
                jobList_node  = [node for node in self.arrivNodeRealization[self.ts_repeat]]
            else:
                jobList       = []
                jobList_wdlay = []
                jobList_node  = []
        else:
            numNewJobs = np.random.poisson(self.arrivalRate)
            jobList = [int(self.newJobSize(self.jobSessionLength)) for ii in range(numNewJobs)]


            ## We generate the wireless latency for each arriving job. 
            ## We add 2 extra ms to account for the frame generation in 5G. 
            ### x1000 to convert from seconds to milliseconds

            jobList_wdlay = [
                (self.FRAME_SIZE / random.uniform(int(self.min_datarate), int(self.max_datarate))) * 1000 + 2 for ii in
                range(numNewJobs)]

            # Jobs can come from any node. randomly associated.
            jobList_node = [int(random.randint(0, self.N_SERVERS - 1)) for ii in range(numNewJobs)]


        return jobList, jobList_wdlay, jobList_node


################################################################################
################################################################################
################################################################################
# %%  ---- AUXILIAR CLASSES FOR LEARNING ------------------------------------ ##
################################################################################
################################################################################
################################################################################

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    # def __init__(self, check_freq: int, log_dir: str, filename = 'best_model', verbose: int = 2):
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 2, model_file_name='best_model'):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        # self.save_path = os.path.join(log_dir, filename)
        self.save_path = os.path.join(log_dir, model_file_name)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward

            """
            ts2xy
            Decompose a data frame variable to x ans ys
            :param data_frame: the input data
            :param x_axis: the axis for the x and y output
                (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
            :return: the x and y output
            """

            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            # print("-------X AND Y----------", x,y)
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                # mean_reward = np.mean(y[-10:])
                mean_reward = np.mean(y[-50:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True
        # Create log dir


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
