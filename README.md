# Green Edge Computing for Latency-Sensitive Applications under Intermittent Renewable Energy

## About the Project
This repository contains the simulation code and environments for the research paper **"Green Edge Computing for Latency-Sensitive Applications under Intermittent Renewable Energy"**. 

The project focuses on Mobile Augmented Reality (MAR) offloading in edge networks powered by intermittent renewable sources. It addresses the fundamental challenges of balancing service quality, operational costs, and environmental impact (carbon footprint) when allocating and migrating compute-intensive, low-latency jobs.

The main approaches evaluated are:
- **GreenRL**: A Deep Reinforcement Learning (DRL) algorithm based on Asynchronous Advantage Actor Critic (A2C) that dynamically allocates and migrates jobs according to the presence of renewable energy.
- **GreenH**: A highly effective heuristic approach adapting to varying network conditions and intermittent energy availability.
- **CarbonEdge**: A carbon-aware workload placement framework for edge computing scenarios [1]. 
- **CC23**: A state-of-the-art algorithm proposed in [2], which also focuses on allocating and migrating jobs using DRL.
- **Solver**:  Optimal solution obtained by solving the corresponding optimization problem.
- **Random**: Simple heuristic, it randomly selects the node to which each job is sent.
- **Emptier**: It sends the job to the node that has the lowest load among the nodes.

## Repository Structure
- `/experiments_parameters/`: YAML configuration files containing simulation parameters for different scenarios (e.g., node counts, load ratios, wireless rates) or for the overall simulation.
- `/plotting/`: Folder containing some files for plotting figures.
- `/test_experiments/`: Folder containing the trained models. Some trained models are present for reproducibility.
- `auxiliarFunctions.py`: File containing helper functions
- `main.py`: The main entry point to run simulations.
- `Solver.py`: Exact/Heuristic Optimization Solver Wrapper based on SCIP, employing a Receding Horizon (Sliding Window) approach to solve job allocation, migration, and power dispatching optimization problems.
- `carbonedge.py`: File containing the code for CE25 [1]
- `energy_datasets.py`: File containing functions to retrieve energy values from ELIA datasets 
- `environ.py`: File containing the environment of GreenRL 
- `environ_sota.py`: File containing the environment of the sota (CC23) [2]
- `evaluation_algorithms.py`: File for evaluating all algorithms expect CC23
- `evaluation_sota.py`: File for evaluating CC23
- `solar.csv`: Renewable source dataset (Solar)
- `wind.csv`: Renewable source dataset (Wind)


## Prerequisites
Ensure you have the following dependencies installed:
- `python3` (>= 3.8)
- `numpy`
- `matplotlib`
- `stable-baselines3` (for GreenRL DRL models)
- `pyscipopt` (SCIP Optimization Suite for Python)

**Important Note for SCIP:** While `pyscipopt` is listed in the Python requirements, it requires the underlying SCIP Optimization Suite to be installed on your operating system first. Please refer to the [official SCIP installation guide](https://scipopt.org/index.php#download) before installing the Python package to avoid compilation errors.

Once SCIP is installed on your system, you can install all Python dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run the Simulations

### Standard Single Run
To start the standard 7-node simulation, use the following command:
```bash
python3 main.py ./experiments_parameters/configuration.yml ./experiments_parameters/7_nodes_experiment_standard_values.yml
```

### Understanding Configuration Files
The execution command relies on two main YAML files to define the simulation behavior:

1. Simulation Control (configuration.yml): This file contains all the top-level configuration parameters to run experiments (e.g., which algorithms to train or evaluate, which objective function to use).
For instance, to train and evaluate the GreenRL algorithm using the fairness function (P2), you would set:

```yaml
GREENRL_TRAINING: 1
POLICY_EVALUATION: 1
REVCOST_F: 0
```

2. Scenario Parameters (e.g., 7_nodes_experiment_standard_values.yml): This file contains the specific data for running a certain scenario, such as the number of edge nodes, network parameters, job parameters, and the names of the trained models to load. 
(Note: For the evaluation of trained algorithms like GreenRL and CC23, you must specify the exact names of the trained models you wish to use inside this file).

### Outputs & Plotting 

When you train the DRL algorithms (GreenRL and CC23), their models will be saved in their respective directories. During the evaluation phase, the final results are compiled and saved into a results_eval.pkl file (or results_solver.pkl for the exact solver) inside the model_policy folder.
To visualize these results, you can use the scripts provided in the /plotting/ folder. For example, to generate the bar plots comparing the algorithms, run:

```bash
python3 plotting/plot_bars.py
```


### Experiment Parameter Files (`/experiments_parameters`)

1. **Base Experiments (Scalability & Standard Values):**
   * `3_nodes_experiment_standard_values.yml`: Defines the baseline configurations for the edge environment, establishing standard arrival rates and server limits.


2. **High Load Experiments:**
   * `5_nodes_experiment_standard_values_high_load_ratio_1.yml`: Evaluates network stress and system resilience by significantly increasing the rate of arriving jobs.

3. **Revenue Ratio Experiments:**
   Adjusts the `REVENUE_FACTOR` and `COST_FACTOR` to assess how the system prioritizes tasks and energy usage under different profitability margins.
   * `3_nodes_experiment_standard_values_ratio_05.yml`: Low profitability scenario (Revenue factor: 10, Cost factor: 20).
   * `3_nodes_experiment_standard_values_ratio_13.yml`: Medium profitability scenario (Revenue factor: 13, Cost factor: 10).
   * `3_nodes_experiment_standard_values_ratio_2.yml`: High profitability scenario (Revenue factor: 10, Cost factor: 5).

4. **Rho (Fairness & Carbon Footprint) Experiments:**
   Varies the `rho_r` parameter in the proportional-fairness objective function to shift the model's focus between maximizing pure profit and saving non-renewable energy.
   * `3_nodes_experiment_standard_values_rho_r_01.yml`: `rho_r` set to 0.1.
   * `3_nodes_experiment_standard_values_rho_r_03.yml`: `rho_r` set to 0.3.
   * `3_nodes_experiment_standard_values_rho_r_05.yml`: `rho_r` set to 0.5.
   * `3_nodes_experiment_standard_values_rho_r_07.yml`: `rho_r` set to 0.7.
   * `3_nodes_experiment_standard_values_rho_r_09.yml`: `rho_r` set to 0.9.

5. **Teraflops & Wireless Rate Experiments:**
   Tests system performance for a 7-node scenario under different MEC server capacities and varied wireless transmission rates. *(Examples of available configurations below)*:

| Configuration File | Server Capacity | Max Wireless Rate | Min Wireless Rate |
| :--- |:----------------|:------------------|:--------|
| `7_nodes_experiment_standard_values_TF_243_WIR_180.yml` | 243 TFLOPS      | 270 Mbps          | 90 Mbps |
| `7_nodes_experiment_standard_values_TF_243_WIR_200.yml` | 243 TFLOPS      | 300 Mbps          | 100 Mbps |
| `7_nodes_experiment_standard_values_TF_328_WIR_100.yml` | 328 TFLOPS      | 150 Mbps          | 50 Mbps |


## References
- **[1]** Li Wu, Walid A. Hanafy, Abel Souza, Khai Nguyen, Jan Harkes, David Irwin, Mahadev Satyanarayanan, and Prashant Shenoy. 2025. CarbonEdge: Leveraging Mesoscale Spatial Carbon-Intensity Variations for Low Carbon Edge Computing. In Proceedings of the 34th International Symposium on High-Performance Parallel and Distributed Computing (HPDC '25). Association for Computing Machinery, New York, NY, USA, Article 12, 1–13. https://doi.org/10.1145/3731545.3731576
- **[2]** E. Karimi, Y. Chen, and B. Akbari, “Task offloading in vehicular edge computing networks via deep reinforcement learning,” Comput. Commun., vol. 189, pp. 193–204, 2022


## Citation
If you find this code useful in your research, please consider citing our paper:
```bibtex
@article{spinelli2026greenedge,
  title={Green Edge Computing for Latency-Sensitive Applications under Intermittent Renewable Energy},
  author={Spinelli, Francesco and Bazco-Nogueras, Antonio and Mancuso, Vincenzo},
  journal={IEEE Open Journal of the Communications Society},
  year={2026},
  publisher={IEEE}
}
```



