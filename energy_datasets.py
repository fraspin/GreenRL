import time
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plot_energy = False  # Set to True to enable plotting

_node_start_info = {}  # key: node index, value: (random_day, start_quarter)

def GreenEnergyGenerationDataset(N_SERVERS, server_num_steps_green, n_time):
    global _node_start_info  

    # ---------------------- Parameters ---------------------- #
    accuracy_improv_wind = 0.4  # parameter to reduce wind variability
    regularizeFactor     = 1.5  # factor to increase wind contribution

    # ---------------------- Load Data ---------------------- #
    T_wind = pd.read_csv('wind_power.csv', sep=';')
    T_solar = pd.read_csv('solar_power.csv', sep=';')

    # ---------------------- Extract Forecast Data ---------------------- #
    forecast_power_wind_week       = np.flip(T_wind.iloc[92:, 15].values)
    forecast_power_wind_lower_week = ((np.flip(T_wind.iloc[92:, 16].values) - forecast_power_wind_week) * accuracy_improv_wind) + forecast_power_wind_week
    forecast_power_wind_upper_week = ((np.flip(T_wind.iloc[92:, 17].values) - forecast_power_wind_week) * accuracy_improv_wind) + forecast_power_wind_week

    forecast_power_solar_week     = np.flip(T_solar.iloc[92:, 7].values)
    forecast_power_solar_week_low = forecast_power_solar_week * 0.95
    forecast_power_solar_week_up  = forecast_power_solar_week * 1.05

    # Apply regularization factor
    forecast_power_wind_week       *= regularizeFactor
    forecast_power_wind_lower_week *= regularizeFactor
    forecast_power_wind_upper_week *= regularizeFactor

    # Combine into dataset
    forecast_dataset = np.column_stack([
        forecast_power_solar_week_low,
        forecast_power_solar_week,
        forecast_power_solar_week_up,
        forecast_power_wind_lower_week,
        forecast_power_wind_week,
        forecast_power_wind_upper_week
    ])

    # Sum of solar + wind
    forecast_power_sum_week_low = forecast_power_solar_week_low + forecast_power_wind_lower_week
    forecast_power_sum_week     = forecast_power_solar_week     + forecast_power_wind_week
    forecast_power_sum_week_up  = forecast_power_solar_week_up  + forecast_power_wind_upper_week
    
    # ---------------------- Normalize Data (Per-Series) ---------------------- #
    def normalize_series(x):
        x_min = np.min(x)
        x_max = np.max(x)
        if x_max - x_min == 0:
            return np.zeros_like(x)  # Avoid division by zero
        return (x - x_min) / (x_max - x_min)

    # Wind
    forecast_power_wind_week       = normalize_series(forecast_power_wind_week)
    forecast_power_wind_lower_week = normalize_series(forecast_power_wind_lower_week)
    forecast_power_wind_upper_week = normalize_series(forecast_power_wind_upper_week)

    # Solar
    forecast_power_solar_week      = normalize_series(forecast_power_solar_week)
    forecast_power_solar_week_low  = normalize_series(forecast_power_solar_week_low)
    forecast_power_solar_week_up   = normalize_series(forecast_power_solar_week_up)

    # Sum
    forecast_power_sum_week_low    = normalize_series(forecast_power_sum_week_low)
    forecast_power_sum_week        = normalize_series(forecast_power_sum_week)
    forecast_power_sum_week_up     = normalize_series(forecast_power_sum_week_up)
    

    # ---------------------- Random Selection Functions ---------------------- #
    energy_node_list = []
    for idx_server in range(N_SERVERS):

        def select_random_day():
            """Select a random day (0=Monday ... 7=Sunday)"""
            return np.random.randint(0, 7)

        def select_random_start_quarter():
            """
            Select a random starting point of the day between 4 AM and 5 PM.
            Each quarter = 15 min -> 4 AM = 16, 5 PM = 68 (inclusive)
            """
            return np.random.randint(16, 69)

        quarters_per_day = 4 * 24

        if n_time == 0:
            random_day          = select_random_day()
            start_quarter       = select_random_start_quarter()
            start_index         = random_day * quarters_per_day + start_quarter
            green_energy_level  = round(forecast_power_sum_week[start_index]*server_num_steps_green[0])
            if green_energy_level <2:
                green_energy_level = 2
            energy_node_list.append(green_energy_level)
            _node_start_info[idx_server] = (random_day, start_quarter)
        else:
            day, start_quarter  = _node_start_info[idx_server]
            next_index          = day * quarters_per_day + start_quarter + n_time
            green_energy_level  = round(forecast_power_sum_week[next_index]*server_num_steps_green[0])
            if green_energy_level <2:
                green_energy_level = 2
            energy_node_list.append(green_energy_level)
            
    # ---------------------- Plotting ---------------------- #
    if plot_energy:
        total_minutes_one_week = 60 * 24 * 7
        x = np.arange(0, total_minutes_one_week, 15)
        x_ticks = np.arange(0, total_minutes_one_week, 60 * 6)
        x_ticklabels = (x_ticks / 60) % 24

        # Full week data
        plt.figure(figsize=(12, 6))
        plt.plot(x, forecast_power_wind_week, 'b-', linewidth=2, label='Wind')
        plt.plot(x, forecast_power_wind_lower_week, 'b--', linewidth=1.5, label='Wind Lower bound')
        plt.plot(x, forecast_power_wind_upper_week, 'b-.', linewidth=1.5, label='Wind Upper bound')

        plt.plot(x, forecast_power_solar_week, 'r-', linewidth=2, label='Solar')
        plt.plot(x, forecast_power_solar_week_low, 'r--', linewidth=1.5, label='Solar Lower bound')
        plt.plot(x, forecast_power_solar_week_up, 'r-.', linewidth=1.5, label='Solar Upper bound')

        plt.grid(True)
        plt.title('Solar/Wind Forecast', fontsize=17)
        plt.xlabel('Time', fontsize=17)
        plt.ylabel('MW', fontsize=17)
        plt.legend(loc='lower center', ncol=1, fontsize=12)
        plt.xticks(x_ticks, x_ticklabels)
        plt.show()

        # Overimposed day data
        quarters_per_day = 4 * 24  # 15-min intervals per day
        x_day = np.arange(0, 60 * 24, 15)
        x_ticks_day = np.arange(0, 60 * 24, 60 * 2)
        x_ticklabels_day = (x_ticks_day / 60) % 24

        plt.figure(figsize=(12, 6))
        for ii in range(7):
            inds_day = slice(ii * quarters_per_day, (ii + 1) * quarters_per_day)
            plt.plot(x_day, forecast_power_wind_week[inds_day], 'b-', label='Wind' if ii == 0 else "")
            plt.plot(x_day, forecast_power_solar_week[inds_day], 'r-', label='Solar' if ii == 0 else "")
            plt.plot(x_day, forecast_power_sum_week[inds_day], 'k-.', label='Sum' if ii == 0 else "")

        plt.grid(True)
        # plt.title('Solar/Wind Forecast (Daily Overlay)', fontsize=17)
        plt.xlabel('Time (hour)', fontsize=17)
        plt.ylabel('MW', fontsize=17)
        plt.legend(['Wind', 'Solar', 'Total'], loc='best', fontsize=12)
        plt.xticks(x_ticks_day, x_ticklabels_day)
        plt.show()

    return energy_node_list

if __name__ == "__main__":
    num_repetitions = 500
    repetition_total_averages = []  # This will store the total average from each of the 30 runs

    print(f"--- Running {num_repetitions} Repetitions of the Simulation ---")
    # 1. Outer loop for statistical repetitions
    for i in range(num_repetitions):
        all_energy_lists = []  # Re-initialize for each independent repetition

        # 2. Original inner loop for one full simulation run (12 time steps)
        for n_time in range(0, 12):
            energy_node_list = GreenEnergyGenerationDataset(3, [10, 10], n_time)
            all_energy_lists.append(energy_node_list)

        # 3. Convert the single run's data to a NumPy array
        data_array = np.array(all_energy_lists)

        # 4. Calculate the total average for this *single* repetition
        if data_array.size > 0:
            total_average_for_run = np.mean(data_array)
            repetition_total_averages.append(total_average_for_run)

    # --- Final Statistical Analysis (after all 30 runs are complete) ---

    if repetition_total_averages:
        # 5. Calculate the final average across all 30 runs
        final_mean = np.mean(repetition_total_averages)

        # 6. Calculate the 95% confidence interval for that final average
        confidence_level = 0.95
        ci = stats.t.interval(confidence_level,
                              df=len(repetition_total_averages) - 1,
                              loc=final_mean,
                              scale=stats.sem(repetition_total_averages))

        # 7. Print the final, aggregated results
        print("\n--- Final Statistical Results ---")
        print(f"Final Total Average (across {num_repetitions} runs): {final_mean:.2f}")
        print(f"{confidence_level * 100:.0f}% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")

    else:
        print("\nNo data was generated to calculate final metrics.")

