import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# Graph 1: M2 Score Sensitivity to Fuel Weight (Corrected Model)
# =================================================================

# --- Model Parameters ---
# Baseline 3-lap time with no fuel (in seconds)
t_base_m2 = 120  
# Use a non-linear (quadratic) penalty for weight to more realistically 
# model the effect of induced drag.
k_linear = 0.05    # Linear penalty component (proportional to weight)
k_quadratic = 0.065  # Quadratic penalty component (proportional to weight^2)

# --- Data Generation ---
# Creates an array of 100 fuel weights to test, from 0 to 6 lbs.
fuel_weights = np.linspace(0, 6, 100)
# Corrected lap time calculation with the non-linear penalty
lap_times = t_base_m2 * (1 + k_linear * fuel_weights + k_quadratic * fuel_weights**2)
# The rated score is the fuel weight (lbs) divided by the lap time in minutes.
m2_rated_scores = fuel_weights / (lap_times / 60)

# --- Find the Optimal Point ---
# np.argmax finds the index of the highest score in the array.
optimal_index = np.argmax(m2_rated_scores)
# Use the index to find the corresponding fuel weight and score.
optimal_fuel_weight = fuel_weights[optimal_index]
optimal_score = m2_rated_scores[optimal_index]

# --- Plotting ---
# Set a professional plot style.
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Plot the main curve of score vs. weight.
plt.plot(fuel_weights, m2_rated_scores, label='M2 Rated Score', 
         color='royalblue', linewidth=2)
# Add a vertical dashed line to mark the optimal fuel weight.
plt.axvline(x=optimal_fuel_weight, color='crimson', linestyle='--', 
            label=f'Optimal Fuel: {optimal_fuel_weight:.2f} lbs')
# Add a marker at the peak of the curve.
plt.plot(optimal_fuel_weight, optimal_score, 'o', 
         color='crimson', markersize=8)

# --- Formatting ---
plt.title('M2 Score Sensitivity to Fuel Weight', fontsize=16)
plt.xlabel('Fuel Weight (lbs)', fontsize=12)
plt.ylabel('M2 Rated Score (Fuel Weight / Lap Time)', fontsize=12)
plt.legend(fontsize=11)
# Save the figure to a file with 300 DPI resolution.
plt.savefig('m2_sensitivity_plot.png', dpi=300) 
plt.close() # Close the plot to free up memory.


# =================================================================
# Graph 2: M3 Bonus Score Sensitivity to X-1 Vehicle Weight
# =================================================================

# --- Model Parameters ---
bonus_values = [25, 50, 75, 100]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc66']

# --- Data Generation ---
# Creates an array of 100 X-1 weights from a very small number (to avoid division by zero)
# up to the maximum allowable weight of 0.55 lbs.
x1_weights = np.linspace(0.05, 0.55, 100)

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Loop through each bonus value and plot its corresponding curve.
for bonus, color in zip(bonus_values, colors):
    m3_bonus_component = bonus / x1_weights
    plt.plot(x1_weights, m3_bonus_component, 
             label=f'{bonus}-Point Box', color=color, linewidth=2.5)

# --- Formatting ---
plt.title('M3 Bonus Score Sensitivity to X-1 Vehicle Weight', fontsize=16)
plt.xlabel('X-1 Vehicle Weight (lbs)', fontsize=12)
plt.ylabel('M3 Bonus Component (Bonus Score / X-1 Weight)', fontsize=12)
# Set a y-axis limit to make the curves easier to compare.
plt.ylim(0, 2100)
plt.legend(fontsize=11)
# Save the second figure to a file.
plt.savefig('m3_sensitivity_plot.png', dpi=300) 
plt.close()

print("Plots 'm2_sensitivity_plot.png' and 'm3_sensitivity_plot.png' have been generated and saved.")
