import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def get_gCO2perRealization(total_brown_power):
    max_pw_server = 10
    polluting_percentOfTotalPw = np.sum(total_brown_power) / max_pw_server
    TS_in_hours = 10 / 3600
    Watt_Server = 900
    total_kwh = TS_in_hours * Watt_Server * polluting_percentOfTotalPw
    gco2e = 372 * total_kwh
    return gco2e


def get_gCO2perSessionOK(results_data, interruptions=False):
    """Calculates Carbon footprint per session."""
    if not results_data:
        return np.nan

    n_realizations = len(results_data)
    data_gCO2e = []
    name_attribute = 'pw_nt_br_sol'

    for n_time in range(n_realizations):
        brown_power = results_data[n_time].get(name_attribute, [0])
        value_gCO2e = get_gCO2perRealization(brown_power)
        data_gCO2e.append(value_gCO2e)

    def getUsersPerType(data, attribute):
        return np.sum([data[idx_realiz].get(attribute, 0) for idx_realiz in range(len(data))])

    j_accepted = getUsersPerType(results_data, 'accepted_jobs_t')
    j_interrup = 0
    if interruptions:
        j_interrup = (getUsersPerType(results_data, 'interruptedpower_t') +
                      getUsersPerType(results_data, 'interrupted_rejected_t') +
                      getUsersPerType(results_data, 'interruptedelay_t'))

    data_users = j_accepted - j_interrup
    if data_users <= 0:
        return 0
    return np.sum(data_gCO2e) / data_users


def get_pareto_frontier(costs):
    """
    Find the pareto-efficient points.
    costs: N x 2 array where col 0 is Carbon (minimize), col 1 is Utility (maximize).
    """
    if costs is None or len(costs) == 0:
        return None

    # Sort by Carbon (ascending) then Utility (descending)
    sorted_indices = np.lexsort((-costs[:, 1], costs[:, 0]))
    sorted_costs = costs[sorted_indices]

    frontier = []
    max_utility = -np.inf

    for cost in sorted_costs:
        # If utility is strictly greater than the max seen so far, it's on the frontier
        if cost[1] > max_utility:
            frontier.append(cost)
            max_utility = cost[1]

    return np.array(frontier)


def plot_pareto_analysis(experiment_dirs, output_dir="test_experiments/pareto_results"):
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # --- LaTeX and Font Size Configuration ---
    # ---------------------------------------------------------
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.fontsize'] = 15
    # ---------------------------------------------------------

    data_rl_profit, data_h_profit, data_opt_profit = [], [], []
    data_rl_fair, data_h_fair, data_opt_fair = [], [], []

    for index, exp_dir in enumerate(experiment_dirs):
        results_eval_file = os.path.join(exp_dir, 'results_eval.pkl')
        results_solver_file = os.path.join(exp_dir, 'results_solver.pkl')

        # Load GreenRL and GreenH
        with open(results_eval_file, 'rb') as mfile:
            results_eval = pickle.load(mfile)  # GreenRL
            _ = pickle.load(mfile)  # env
            _ = pickle.load(mfile)  # random
            _ = pickle.load(mfile)  # emptier
            results_eval_heuristic = pickle.load(mfile)  # GreenH

        # Load Optimal
        try:
            with open(results_solver_file, 'rb') as mfile:
                results_solver = pickle.load(mfile)
        except (FileNotFoundError, EOFError):
            results_solver = None

        # --- Extract Profit Utilities ---
        profit_rl = np.average([res['Objective_function_rev_minus_cost'] for res in results_eval])
        profit_h = np.average([res['Objective_function_rev_minus_cost'] for res in results_eval_heuristic])
        profit_opt = np.average(
            [res['Objective_function_rev_minus_cost'] for res in results_solver]) if results_solver else np.nan

        # --- Extract Fairness Utilities ---
        fair_rl = np.average([res['objFunction_Fairness'] for res in results_eval])
        fair_h = np.average([res['objFunction_Fairness'] for res in results_eval_heuristic])
        fair_opt = np.average([res['objFunction_Fairness'] for res in results_solver]) if results_solver else np.nan

        # --- Extract Carbon ---
        carbon_rl = get_gCO2perSessionOK(results_eval, interruptions=True)
        carbon_h = get_gCO2perSessionOK(results_eval_heuristic, interruptions=True)
        carbon_opt = get_gCO2perSessionOK(results_solver, interruptions=False) if results_solver else np.nan

        # Append to Profit data arrays
        data_rl_profit.append([carbon_rl, profit_rl])
        data_h_profit.append([carbon_h, profit_h])
        if results_solver:
            data_opt_profit.append([carbon_opt, profit_opt])
            data_opt_fair.append([carbon_opt, fair_opt])

        # Append to Fairness data arrays
        data_rl_fair.append([carbon_rl, fair_rl])
        data_h_fair.append([carbon_h, fair_h])

    # Convert to Numpy Arrays
    data_rl_profit, data_h_profit = np.array(data_rl_profit), np.array(data_h_profit)
    data_rl_fair, data_h_fair = np.array(data_rl_fair), np.array(data_h_fair)
    data_opt_profit = np.array(data_opt_profit) if data_opt_profit else None
    data_opt_fair = np.array(data_opt_fair) if data_opt_fair else None

    # Calculate Frontiers
    front_rl_p = get_pareto_frontier(data_rl_profit)
    front_h_p = get_pareto_frontier(data_h_profit)
    front_opt_p = get_pareto_frontier(data_opt_profit)

    front_rl_f = get_pareto_frontier(data_rl_fair)
    front_h_f = get_pareto_frontier(data_h_fair)
    front_opt_f = get_pareto_frontier(data_opt_fair)

    # ---------------------------------------------------------
    # --- Plot 1: Profit vs Carbon ---
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.scatter(data_rl_profit[:, 0], data_rl_profit[:, 1], color='green', alpha=0.3)
    plt.scatter(data_h_profit[:, 0], data_h_profit[:, 1], color='blue', alpha=0.3)
    if data_opt_profit is not None:
        plt.scatter(data_opt_profit[:, 0], data_opt_profit[:, 1], color='black', alpha=0.3)

    plt.plot(front_rl_p[:, 0], front_rl_p[:, 1], 's-', color='green', label='GreenRL', linewidth=2)
    plt.plot(front_h_p[:, 0], front_h_p[:, 1], '^-', color='blue', label='GreenH', linewidth=2)
    if front_opt_p is not None:
        plt.plot(front_opt_p[:, 0], front_opt_p[:, 1], 'o-', color='black', label='Optimal', linewidth=2)

    plt.xlabel(r'gCO$_{2}$e per MAR session [$\rightarrow$ Better]')
    plt.ylabel(r'Profit margin ($\bar{B}$) [$\rightarrow$ Better]')
    # Removing title for standard LaTeX inclusion, add back if needed
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    profit_save_path = os.path.join(output_dir, 'pareto_frontier_profit.pdf')
    plt.savefig(profit_save_path)
    print(f"Profit Pareto frontier saved to {profit_save_path}")
    plt.close() # Close the figure to start a fresh one

    # ---------------------------------------------------------
    # --- Plot 2: Fairness vs Carbon ---
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.scatter(data_rl_fair[:, 0], data_rl_fair[:, 1], color='green', alpha=0.3)
    plt.scatter(data_h_fair[:, 0], data_h_fair[:, 1], color='blue', alpha=0.3)
    if data_opt_fair is not None:
        plt.scatter(data_opt_fair[:, 0], data_opt_fair[:, 1], color='black', alpha=0.3)

    plt.plot(front_rl_f[:, 0], front_rl_f[:, 1], 's-', color='green', label='GreenRL', linewidth=2)
    plt.plot(front_h_f[:, 0], front_h_f[:, 1], '^-', color='blue', label='GreenH', linewidth=2)
    if front_opt_f is not None:
        plt.plot(front_opt_f[:, 0], front_opt_f[:, 1], 'o-', color='black', label='Optimal', linewidth=2)

    plt.xlabel(r'gCO$_{2}$e per MAR session [$\rightarrow$ Better]')
    plt.ylabel(r'Fairness $\log(\bar{P}_{\rho_p}\bar{R}_{\rho_r})$ [$\rightarrow$ Better]')
    # plt.title('Pareto Frontier: Fairness vs. Carbon')
    plt.gca().invert_xaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.legend()
    plt.legend(loc='lower right')
    plt.tight_layout()

    fairness_save_path = os.path.join(output_dir, 'pareto_frontier_fairness.pdf')
    plt.savefig(fairness_save_path)
    print(f"Fairness Pareto frontier saved to {fairness_save_path}")
    plt.close()


if __name__ == "__main__":
    # Point this to your actual experiment directories for varying rho
    exp_folder = os.path.join(ROOT_DIR, 'test_experiments')

    ### PUT HERE THE EXPERIMENT YOU WANT TO PLOT ###
    # Directories where the agent was trained for Fairness
    profit_dirs = [

    ]

    # Directories where the agent was trained for Fairness
    fairness_dirs = [

    ]

    all_experiment_dirs = profit_dirs + fairness_dirs

    full_dirs = [os.path.join(exp_folder, d) for d in all_experiment_dirs]
    output_dir = os.path.join(exp_folder, "pareto_results")
    plot_pareto_analysis(full_dirs,output_dir=output_dir)