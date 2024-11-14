import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def calculate_heat_capacity(frequency, phonon_DOS, N):
    # Constants
    h_bar = 1.0546e-34  # Planck's constant (Js)
    kB = 1.38065e-23    # Boltzmann constant (J/K)

    # Heat capacity calculation function
    def heat_capacity(omega, dos, T):
        x = h_bar * omega / (kB * T)
        x_clipped = np.clip(x, None, 100)  # Prevent overflow
        exp_x = np.exp(x_clipped)
        expm1_x = np.expm1(x_clipped)
        expm1_x_sq = np.where(expm1_x == 0, np.inf, expm1_x**2)
        integrand = (x_clipped**2 * exp_x / expm1_x_sq) * dos
        Cv = kB * np.trapz(integrand, omega)
        return Cv

    # Adjustment factor
    target_heat_capacity = 2078  # J/(K·kg)
    temperature_high = 5000      # K
    Cv_original = heat_capacity(frequency, phonon_DOS, temperature_high)
    adjustment_factor = target_heat_capacity / Cv_original

    # Temperature range and heat capacity calculation
    temperatures = 10 ** np.linspace(0, 3.2, 200)
    heat_capacities = [heat_capacity(frequency, phonon_DOS, T) * adjustment_factor for T in temperatures]

    # Heat capacity fitting function at low temperatures
    def cv_temp_relation(T, n, a):
        return a * T ** n

    # Define new fitting ranges
    fit_ranges = [
        ('1-5 K', {'min_temp': 1, 'max_temp': 5, 'color': 'r'}),
        ('5-10 K', {'min_temp': 5, 'max_temp': 10, 'color': 'g'}),
        ('10-100 K', {'min_temp': 10, 'max_temp': 100, 'color': 'b'})
    ]

    n_fitted_values = []

    plt.figure(figsize=(8, 6))
    plt.plot(temperatures, heat_capacities, 'k-', linewidth=2, label='Heat Capacity')

    # Perform fitting and plot the fitted curves
    for label, params in fit_ranges:
        min_temp = params['min_temp']
        max_temp = params['max_temp']
        color = params['color']
        indices = np.where((temperatures >= min_temp) & (temperatures <= max_temp))[0]
        temps_fit = temperatures[indices]
        heat_caps_fit = np.array(heat_capacities)[indices]

        # Fit Cv = a * T^n
        popt, _ = curve_fit(cv_temp_relation, temps_fit, heat_caps_fit)
        n_fitted, a_fitted = popt
        n_fitted_values.append(n_fitted)

        # Plot the fitted curve
        plt.plot(temps_fit, cv_temp_relation(temps_fit, n_fitted, a_fitted), f'{color}--', linewidth=2,
                 label=f'{label}: $C_v \\sim T^{{{n_fitted:.2f}}}$')
        print(f'Fitted exponent n in the range {label}: {n_fitted:.2f}')

    plt.xlabel('$T$, K', fontsize=18)
    plt.ylabel('$C_v$, J/(K·kg)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return temperatures, heat_capacities, n_fitted_values
