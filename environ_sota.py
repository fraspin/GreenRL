import time

import numpy as np

import random
import gymnasium
#import gym
from sklearn.metrics.pairwise import euclidean_distances

#from gym import spaces
from gymnasium import spaces
from energy_datasets import GreenEnergyGenerationDataset


class WorldEnvStateOfTheArt(gymnasium.Env):
    """
    Custom Environment following the Gymnasium interface for the State-of-the-Art (SOTA) algorithm.
    This environment represents the CC23 benchmark algorithm, which focuses primarily on
    task delays (access, queue, computation, and migration) and strict deadlines.
    """

    def reset(self, seed=None, repeat=False, jobs=[], greenpw=[], propagationDelay=[],
              jobs_wdlay=[], jobs_arrivalNode=[],isTraining=True):

        """
        Resets the environment to its initial state at the beginning of an episode.

        Args:
            seed (int): Random seed for reproducibility.
            repeat (bool): If True, repeats a specific deterministic realization (for evaluation).
            jobs, greenpw, propagationDelay, jobs_wdlay, jobs_arrivalNode: Data arrays used if repeat is True.
            isTraining (bool): Dictates whether to use random energy generation or load from a dataset.

        Returns:
            current_state_dict (dict): The initial observation mapped to the Dict space.
            info (dict): Empty dictionary required by Gymnasium.
        """

        self.TIME_SLOT = 0

        # Initialize the current and next state dictionaries
        # Representing node queues, node computation loads, and incoming task metadata
        self.current_state_dict = {}
        self.current_state_dict = {"nodes_comp_queue": np.zeros(self.N_SERVERS, dtype=int),
                                   "nodes_stor_queue": np.zeros(self.N_SERVERS, dtype=int),
                                   "nodes_comp_nodes": np.zeros(self.N_SERVERS, dtype=int),
                                   "nodes_stor_nodes": np.zeros(self.N_SERVERS, dtype=int),
                                   "task_comp": np.zeros(1, dtype=int),
                                   "task_stor": np.zeros(1, dtype=int),
                                   "task_deadline": np.zeros(1, dtype=int),
                                   "task_node_location": np.zeros(1, dtype=int)
                                   }

        self.next_state_dict = {}
        self.next_state_dict = {"nodes_comp_queue": np.zeros(self.N_SERVERS, dtype=int),
                                "nodes_stor_queue": np.zeros(self.N_SERVERS, dtype=int),
                                "nodes_comp_nodes": np.zeros(self.N_SERVERS, dtype=int),
                                "nodes_stor_nodes": np.zeros(self.N_SERVERS, dtype=int),
                                "task_comp": np.zeros(1, dtype=int),
                                "task_stor": np.zeros(1, dtype=int),
                                "task_deadline": np.zeros(1, dtype=int),
                                "task_node_location": np.zeros(1, dtype=int)
                                }

        self.reward = 0
        self.ts_repeat = 0

        self.jobs_list = []
        self.jobs_list_delay = []

        # Tracking metrics for evaluation
        self.error_migration = 0
        self.lastJobExceededNodePw = 0
        self.newJobExceededNodePw = 0
        self.migratedJobExceededNodePw = 0
        self.lastJobExceededDeadline = 0
        self.lastJobExceededQueue = 0
        self.ActionSpaceExceeded = 0
        self.error_capacity = 0

        # System state tracking lists for nodes and queues
        self.NodeList = [[]] * self.N_SERVERS
        self.jobs_migratedList = [[]] * self.N_SERVERS
        self.DurationList = [[]] * self.N_SERVERS
        self.QueueList = [[]] * self.N_SERVERS
        self.QueueDurationList = [[]] * self.N_SERVERS
        self.DelayList = [[]] * self.N_SERVERS

        self.migration = np.zeros(self.N_SERVERS)
        self.migration_sota = 0

        ## Index of the current job
        self.jobIndex = 0

        ## Binary variable to set if we are starting a new Time Slot
        self.isNewTS = 1

        ### Saving if we are repeating the environment (baselines) or not
        self.repeatRealization = repeat

        # Generate or retrieve green energy values based on training/testing phase
        if self.repeatRealization == False:

            if isTraining: #if we are training, we generate random green energy values
                self.greenPWrealization = self.greenEnergyGeneration()
            else: # if we are not training, we load pre-generated green energy values from datasets
                self.greenPWrealization = GreenEnergyGenerationDataset(self.N_SERVERS,self.MAX_PW_SERVERS, 0)
        else:
            self.greenPWrealization = greenpw[:, 0]
            self.green_power_levels = self.greenPWrealization

        # Setup Node Distances and Delays
        if self.repeatRealization:
            self.jobsRealization = jobs
            self.wlessDelayRealization = jobs_wdlay
            self.arrivNodeRealization = jobs_arrivalNode
            self.matrix_propagation_delay = propagationDelay[:, :]

        else:  # if self.repeatRealization == False:
            ## Generating random location of servers
            self.serverLocations = np.random.uniform(size=[self.N_SERVERS, 2]) * self.SIZE_KM_SETTING_SIDE
            self.serversDistances = euclidean_distances(self.serverLocations, self.serverLocations)
            self.matrix_propagation_delay = self.serversDistances * 5e-3  # speed og light fiber = 200 m/micro-sec -> 5 microsec/km -> 0.005 ms/km

        self.jobs_list, self.jobs_list_delay, self.job_list_arrivNode = self.generateNewJobs()

        self.current_state_dict = self.getUpdateNextState(self.current_state_dict)
        info = {}
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
        Initialization of the SOTA Environment. Maps all hyperparameters required for simulation.
        """
        super(WorldEnvStateOfTheArt, self).__init__()

        # Heavy penalty for invalid actions or constraint violations
        self.penalty = 10

        # Server and Network Parameters
        self.N_SERVERS = num_servers  ## Number of edge servers/nodes
        self.fiber_datarate = fiberDataRate
        self.min_datarate = minWirelessRate
        self.max_datarate = maxWirelessRate
        self.FRAME_SIZE = frameSize

        # Delay Constraints
        self.COMPUTATION_DELAY = computationDelay
        self.SWITCHING_DELAY = switchingDelay
        self.DEADLINE_DELAY = deadlineDelay
        self.TRASMISSION_DELAY = (self.FRAME_SIZE / self.fiber_datarate) * 1000 + self.SWITCHING_DELAY  # in ms

        # Economic / Reward Factors
        self.COST_FACTOR = cost_factor
        self.REV_FACTOR = revenue_factor  # proportial coefficient for revenues
        self.arrivalRate = arrivalRate
        self.jobSessionLength = jobSessionLength  # Number of jobs in the system - useful for defining action space and observation space
        self.JOB_STORAGE = job_storage

        # Ensure capacities are lists mapped to servers
        if not isinstance(max_pc_server, list):
            self.MAX_PC_SERVERS = [max_pc_server] * self.N_SERVERS
        else:
            self.MAX_PC_SERVERS = max_pc_server

        # Increment by 1 to accommodate the Discrete space definition
        # (spaces.MultiDiscrete is exclusive of the upper bound)
        self.MAX_PC_SERVERS_plus = [value + 1 for value in self.MAX_PC_SERVERS]

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

        self.matrix_propagation_delay = np.zeros((self.N_SERVERS, self.N_SERVERS), dtype=int)

        self.jobsRealization = []
        self.wlessDelayRealization = []
        self.arrivNodeRealization = []

        self.c_jn = c_jn

        # Objective Function Weights and Parameters
        self.alpha, self.beta, self.gamma = pw_params  # tuple of 3 values of proc-cycles to power conversor
        self.rho_d, self.rho_p, self.rho_r = obj_params  # factors to weight delay, power, and revenue

        self.T_buffer = T_buffer
        self.window_m = window_m

        self.revcost_function = revCost_function  # 1 if profit, 0 if fairness
        self.noLog_fairness = nolog_fairness  # 1 if no log in fairness, 0 if log

        # --- DEFINE ACTION SPACE ---
        # Action space: 0 = Reject, 1 to N_SERVERS = Accept at node X.
        self.action_space = spaces.Discrete(self.N_SERVERS + 1)

        # --- DEFINE OBSERVATION SPACE ---
        self.buffer = [1] * self.N_SERVERS  ### Queue is either 0 or 1

        if self.DEADLINE_DELAY == 33:
            index_deadline = 0
        else:
            # Multi-deadline configurations can be enabled here
            index_deadline = 1

        # Evaluate if storage is a strict constraint for this experiment
        if (self.JOB_STORAGE / self.MAX_STORAGE_SERVER[0]) * 100 > 10:
            index_storage = self.N_SERVERS * [1]
            index_storage_job = 1
        else:
            index_storage = np.zeros(self.N_SERVERS, dtype=int)
            index_storage_job = 0

        index_storage_adjusted = [value + 1 for value in index_storage]

        quantized_node = [2] * self.N_SERVERS

        # Dictionary observation space explicitly tracking queues, loads, and incoming task specs
        self.observation_space = spaces.Dict({
            "nodes_comp_queue": spaces.MultiDiscrete(np.array(self.buffer, dtype=np.int32)),
            "nodes_stor_queue": spaces.MultiDiscrete(np.array(self.buffer, dtype=np.int32)),
            "nodes_comp_nodes": spaces.MultiDiscrete(np.array(self.MAX_PC_SERVERS_plus, dtype=np.int32)),
            "nodes_stor_nodes": spaces.MultiDiscrete(np.array(index_storage_adjusted, dtype=np.int32)),
            "task_comp": spaces.Discrete(1),
            "task_stor": spaces.Discrete(index_storage_job + 1),
            "task_deadline": spaces.Discrete(index_deadline + 1),
            "task_node_location": spaces.Discrete(self.N_SERVERS)
        })

        self.reset()

    # --- Delay Formulation Helpers (Based on SOTA Reference Paper CC23) ---

    def getTauAccess(self, obs_space):
        """
        Calculates the access delay (wireless link latency).
        Note: The referenced paper considers RTT instead of one-way latency, requiring
        adjustments in the main step logic to ensure fair comparisons.
        """

        return self.jobs_list_delay[0]

    def getTauQueue(self, obs_space, action):
        """
        Calculates the queueing delay.
        Currently set to 0 to align with GreenRL assumptions
        """

        return 0

    def getTauComp(self, obs_space):
        """Calculates computation delay (assumed uniform across nodes)."""
        return self.COMPUTATION_DELAY

    def getTauMigration(self, obs_space, action):
        """Calculates the migration delay over fiber links if the task is offloaded."""

        t_mig = self.TRASMISSION_DELAY
        return t_mig

    def step(self, action):
        """
                Main simulation step. Processes one incoming job at a time.

                Logic Flow:
                1. Parse the action (Reject or Allocate to Node X).
                2. If Allocate: Calculate the total delays (Access, Queue, Compute, Migration).
                3. Check against DEADLINE_DELAY constraint.
                4. If valid: Update System state and assign positive reward.
                   If invalid: Assign massive penalty.
                5. If the job list is empty: Fast-forward the Time Slot.

                Args:
                    action (int): The selected action from the discrete space.

                Returns:
                    next_state, reward, done, truncated, info
                """
        done = False
        info = {}

        self.next_state_dict = self.current_state_dict
        self.lastJobExceededDeadline = 0
        self.lastJobExceededQueue = 0
        self.ActionSpaceExceeded = 0
        self.migration_sota = 0

        # Check if there are newly arrived jobs in the queue
        if self.jobs_list != []:

            # Agent selected a valid node to allocate the task (Action 1 to N_SERVERS)
            if action != 0:
                # Calculate latency components
                t_access = self.getTauAccess(self.current_state_dict)
                t_queue = self.getTauQueue(self.current_state_dict, action)
                t_comp = self.getTauComp(self.current_state_dict)

                # Check if the chosen node is the node where the task arrived (Local Execution)
                if (action-1) == int(self.current_state_dict['task_node_location']):
                    # Calculate total Round-Trip Delay for local execution
                    total_tau = (t_access) * 2 + t_queue + t_comp

                    if total_tau <= self.DEADLINE_DELAY:
                        # Constraint satisfied: Commit allocation
                        flag_job_processing_mode = 1
                        done = self.UpdateSystemLists(flag_job_processing_mode, action, done, total_tau)

                        if done == False:
                            self.reward = total_tau * 1.0

                    else:
                        # Deadline missed: Apply penalty
                        self.reward = -self.penalty * total_tau * 1.0
                        self.lastJobExceededDeadline = 1
                        done = True


                else:
                    # Orchestrator chose to migrate the job (Offloading)
                    t_mig = self.getTauMigration(self.current_state_dict, action)
                    # Calculate total Round-Trip Delay including migration
                    total_tau = (t_access + t_mig) * 2 + t_queue + t_comp


                    if total_tau <= self.DEADLINE_DELAY:
                        # Constraint satisfied: Commit allocation and log migration
                        flag_job_processing_mode = 1
                        done = self.UpdateSystemLists(flag_job_processing_mode, action, done, total_tau)
                        self.migration_sota = 1
                        if done == False:
                            self.reward = total_tau * 1.0
                            self.migration[action-1 ] += 1

                    else:
                        self.reward = -self.penalty * total_tau * 1.0
                        self.lastJobExceededDeadline = 1
                        done = True

                # Job has been processed; remove from incoming list
                del self.jobs_list[0]
                del self.jobs_list_delay[0]
                del self.job_list_arrivNode[0]
            else:
                # Action 0 implies the Orchestrator rejected the job
                # The paper mentions rejection but lacks reward mechanics; we assign 0 reward here.

                self.reward = 0

                del self.jobs_list[0]
                del self.jobs_list_delay[0]
                del self.job_list_arrivNode[0]

            # If jobs remain in the list, update observation state for the next step within the same TS
            if self.jobs_list != []:
                self.next_state_dict = self.getUpdateNextState(self.next_state_dict)


        # If all incoming jobs for the current TS have been processed, advance the Time Slot
        if self.jobs_list == []:


            self.ts_repeat += 1
            self.jobs_list, self.jobs_list_delay, self.job_list_arrivNode = self.generateNewJobs()


            # flag_job_processing_mode = 2 triggers the TS update (decrementing active job durations)
            flag_job_processing_mode = 2
            done = self.UpdateSystemLists(flag_job_processing_mode, action, done, 0)

            # Update state with the first job of the new TS
            self.next_state_dict = self.getUpdateNextState(self.next_state_dict)
            self.TIME_SLOT += 1

        self.current_state_dict = self.next_state_dict

        truncated = False

        return self.current_state_dict, self.reward, done, truncated, info


    def UpdateSystemLists(self, flag_job_processing_mode, action, done, total_tau):
        """
                Updates the internal state tracking lists for nodes, jobs, and durations based on the action taken.

                Args:
                    flag_job_processing_mode (int): 1 for Job Acceptance/Allocation, 2 for Time Slot Advance.
                    action (int): The chosen action.
                    done (bool): Terminal state flag.
                    total_tau (float): Total calculated delay (used for penalty scaling).

                Returns:
                    done (bool): Updated terminal state flag if capacity constraints are violated.
        """
        if flag_job_processing_mode == 1:
            # --- ACCEPTANCE & ALLOCATION PHASE ---
            AvailableNodeslist = [int(np.sum(jobs_pc_list)) for jobs_pc_list in self.NodeList]

            chosen_node = action -1

            # Validate Capacity Constraints
            if (AvailableNodeslist[chosen_node] + int(self.c_jn)) > self.MAX_PC_SERVERS[0]:

                # Node exceeds maximum processing capacity
                self.lastJobExceededQueue = 1
                self.lastJobExceededNodePw = 1

                self.reward = -self.penalty * 1.0 * total_tau
                done = True

            else:
                # Node is valid, allocate task directly to the server
                if self.NodeList[chosen_node] == []:
                    self.NodeList[chosen_node] = [int(self.c_jn)]
                    self.DurationList[chosen_node] = [int(self.jobs_list[0])]  ### D_MAX = Duration of job
                else:
                    self.NodeList[chosen_node].append(int(self.c_jn))
                    self.DurationList[chosen_node].append(int(self.jobs_list[0]))


        # --- TIME SLOT ADVANCE PHASE ---
        # Decrement duration of all active jobs. Remove finished jobs.
        if flag_job_processing_mode == 2:
            for node_idx, nodeJobs in enumerate(self.DurationList):
                job_idx_new = 0
                for jobLength in nodeJobs:  # for each job in the node
                    if jobLength != 0:
                        self.DurationList[node_idx][job_idx_new] = jobLength - 1
                        # If job has finished, mark for deletion
                        if self.DurationList[node_idx][job_idx_new] == 0:
                            self.NodeList[node_idx][job_idx_new] = 0
                        job_idx_new += 1
                    else:
                        job_idx_new += 1

                # Filter out terminated jobs
                self.DurationList[node_idx] = [value for value in self.DurationList[node_idx] if value != 0]
                self.NodeList[node_idx] = [value for value in self.NodeList[node_idx] if value != 0]


        return done

    def getUpdateNextState(self, obs_space):
        """
                Constructs the observation dictionary provided to the RL Agent.
                Reflects current node loads, queue statuses, and the features of the NEXT job in line.
        """

        obs_space['task_comp'] = 0
        index = 0

        # Storage is neglected to remain consistent with GreenRL assumptions
        if (self.JOB_STORAGE / self.MAX_STORAGE_SERVER[0]) * 100 > 10:
            index_storage = self.N_SERVERS * [1]
            index_storage_job = 1
        else:
            index_storage = np.zeros(self.N_SERVERS, dtype=int)
            index_storage_job = 0

        obs_space['task_stor'] = int(index_storage_job)
        obs_space['task_deadline'] = int(index)


        # Set arrival location for the new job
        if self.job_list_arrivNode == []:
            obs_space['task_node_location'] = int(random.randint(0, self.N_SERVERS - 1))
        else:
            obs_space['task_node_location'] = self.job_list_arrivNode[0]

        # Update each node's load state within the observation dictionary
        for node_idx, nodeJobs in enumerate(self.NodeList):
            """
            Update observation space with new values regarding the environment
            """
            if self.NodeList[node_idx] != []:
                update_node = int(np.sum(self.NodeList[node_idx]))
            else:
                update_node = int(0)

            # Queues fixed to 0 in this implementation
            obs_space['nodes_comp_queue'][node_idx] = 0
            obs_space['nodes_stor_queue'][node_idx] = 0
            obs_space['nodes_comp_nodes'][node_idx] = update_node
            obs_space['nodes_stor_nodes'][node_idx] = index_storage[0]

        return obs_space

    # --- Energy and Job Generation Functions ---

    def updatePcState(self):
        """Updates internal raw tracking vector with total processing cycles per node."""
        self.next_state_raw[:self.N_SERVERS] = [int(np.sum(jobs_pc_list)) for jobs_pc_list in self.NodeList]

    def getPowerConsumption(self):
        """Calculates energy consumption based on node load."""
        power = np.zeros(self.N_SERVERS)
        for nodeIdx, nodeJobs in enumerate(self.NodeList):
            for jobPC in nodeJobs:
                power[nodeIdx] += (self.alpha * jobPC + self.gamma)

        return power

    def greenEnergyGeneration(self, current_energy=[], node=None):
        """Generates random green energy limits (used primarily during training)."""
        if node == None:  ## return full vector
            new_energy = [round(random.randint(2, self.MAX_PW_SERVERS[node])) for node in range(self.N_SERVERS)]
        else:  ## return just one server
            new_energy = round(random.randint(2, self.MAX_PW_SERVERS[node]))
        return new_energy

    def newJobSize(self, max_cycles):
        """Returns length in Time Slots for an arriving job."""
        return self.jobSessionLength

    def generateNewJobs(self):
        """
                Generates incoming jobs using a Poisson process for arrivals,
                assigning random wireless delays and arrival nodes.
        """
        if self.repeatRealization:
            # Reproduce identical job traces during evaluation
            if self.ts_repeat < len(self.jobsRealization):
                jobList = [job for job in self.jobsRealization[self.ts_repeat]]
                jobList_wdlay = [job_wd for job_wd in self.wlessDelayRealization[self.ts_repeat]]
                jobList_node = [node for node in self.arrivNodeRealization[self.ts_repeat]]
            else:
                jobList = []
                jobList_wdlay = []
                jobList_node = []
        else:
            # Generate new stochastic workloads for training
            numNewJobs = np.random.poisson(self.arrivalRate)

            jobList = [int(self.newJobSize(self.jobSessionLength)) for ii in range(numNewJobs)]

            # Generate wireless latency (+2ms static overhead for 5G frame generation)
            jobList_wdlay = [
                (self.FRAME_SIZE / random.randint(int(self.min_datarate), int(self.max_datarate))) * 1000 + 2 for ii in
                range(numNewJobs)]

            # Randomize Arrival Node
            jobList_node = [int(random.randint(0, self.N_SERVERS - 1)) for ii in range(numNewJobs)]


        return jobList, jobList_wdlay, jobList_node

