# main.py

from dispersion import compute_dispersion
from heat_capacity import calculate_heat_capacity
import matplotlib.pyplot as plt
import numpy as np
import csv


def export_n_fitted_data(N_values, n_fitted_array, GNR_type):
    # Export n_fitted data to a CSV file
    with open(f'n_fitted_vs_N_{GNR_type}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'n_fitted_1-5K', 'n_fitted_5-10K', 'n_fitted_10-100K'])
        for i, N in enumerate(N_values):
            writer.writerow([N, n_fitted_array[i, 0], n_fitted_array[i, 1], n_fitted_array[i, 2]])


def export_heat_capacity_data(N_values, all_temperatures, all_heat_capacities, GNR_type):
    # Export heat capacity data to a CSV file
    with open(f'heat_capacity_vs_temperature_{GNR_type}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['N', 'Temperature', 'Heat_Capacity'])
        for i, N in enumerate(N_values):
            for j in range(len(all_temperatures[i])):
                writer.writerow([N, all_temperatures[i][j], all_heat_capacities[i][j]])


def main():
    N_values = list(range(4, 25))  # List of N values to iterate over
    GNR_type = 'AGNR'  # Graphene nanoribbon type

    # Prepare lists to store temperatures and heat capacities for each N
    all_temperatures = []
    all_heat_capacities = []

    # Line styles to use for different N values
    line_styles = ['-', '--', '-.', ':', '--']  # Four different line styles

    n_fitted_list = []

    for idx, N in enumerate(N_values):
        print(f"Computing for N = {N}...")

        # Compute dispersion relations
        print("Computing dispersion relations...")
        k_list, omega, frequency_bins, phonon_DOS = compute_dispersion(N, GNR_type)

        print(phonon_DOS)

        # Compute heat capacity
        print("Computing heat capacity...")
        temperatures, heat_capacities, n_fitted = calculate_heat_capacity(frequency_bins, phonon_DOS, N)

        # Store the temperature and heat capacity data
        all_temperatures.append(temperatures)
        all_heat_capacities.append(heat_capacities)

        print(f"Fitted exponents for N={N}: {n_fitted}")

        n_fitted_list.append(n_fitted)

    n_fitted_array = np.array(n_fitted_list)
    N_values_array = np.array(N_values)

    # Export data to CSV files
    export_n_fitted_data(N_values_array, n_fitted_array, GNR_type)
    export_heat_capacity_data(N_values_array, all_temperatures, all_heat_capacities, GNR_type)

    print("Data export completed successfully.")

    # Plot n_fitted vs N for each temperature range
    plt.figure(figsize=(8, 6))

    # First line: solid line, circle markers
    plt.plot(N_values_array, n_fitted_array[:, 0],
             marker='o', linestyle='-', color='k', linewidth=2, markersize=8, label='1-5K')

    # Second line: dashed line, square markers
    plt.plot(N_values_array, n_fitted_array[:, 1],
             marker='s', linestyle='--', color='k', linewidth=2, markersize=8, label='5-10K')

    # Third line: dash-dot line, triangle markers
    plt.plot(N_values_array, n_fitted_array[:, 2],
             marker='^', linestyle='-.', color='k', linewidth=2, markersize=8, label='10-100K')

    plt.xlabel('N', fontsize=18)
    plt.ylabel('$n_{fitted}$', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'n_fitted_vs_N_{GNR_type}.png', dpi=600)
    plt.show()

    # Plot heat capacity vs temperature for each N
    plt.figure(figsize=(8, 6))

    for idx, N in enumerate(N_values):
        if idx % 4 == 0:
            plt.loglog(all_temperatures[idx], all_heat_capacities[idx],
                       linestyle=line_styles[idx % len(line_styles)],
                       color='k',
                       linewidth=2,
                       label=f'N={N}')

    plt.xlabel('$T$, K', fontsize=18)
    plt.ylabel('$C_v$, J/(K·kg)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.xlim([1, 100])
    plt.ylim([0.3, 300])
    plt.tight_layout()
    plt.savefig(f'heat_capacity_vs_temperature_loglog_{GNR_type}.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    main()
