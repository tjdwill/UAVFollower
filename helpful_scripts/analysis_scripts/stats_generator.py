# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import chdir
from pathlib import Path


# Initial Config
mpl.rcParams['figure.dpi'] = 300  # Hi-res figures

# Pathing: Load files
chdir(Path("C:/Users/Tj/Documents/GradSchool/Thesis/Programming/ros/20240306_2009_uav_follower_analysis/bags/following_experiments"))

file_exp02 =  Path("exp02/csv/uav_follow_exp_2024-03-05-18-45-24_combined.csv")
file_exp03 = Path("exp03/csv/uav_follow_exp_2024-03-05-19-00-55_combined.csv")
exp_file = file_exp03

# Read the data into the variables
csv = pd.read_csv(exp_file)
columns = csv.columns.values.tolist()

x_cov_offset = 0.247
t, drone_x, drone_y, est_x, est_y ,hexa_x, hexa_y, depth = columns

t = csv[t].values
t = t-t[0]

drone_x = csv[drone_x].values
drone_y = csv[drone_y].values

est_x = csv[est_x].values
est_y = csv[est_y].values

hexa_x = csv[hexa_x].values
# hexa_x[0] += x_cov_offset  # Adjust first x point
hexa_y = csv[hexa_y].values

future_hexa_x = hexa_x[1:]
future_hexa_y = hexa_y[1:]
future_hexa_t = t[0:-1]

depth = csv[depth].values

# %% Report Statistics
error_est_x = est_x - drone_x
error_est_y = est_y - drone_y

follow_dist_est = est_x[0:-1] - future_hexa_x
follow_dist_real = drone_x[0:-1] - future_hexa_x
follow_dist_avg_est = np.average(est_x[0:-1] - future_hexa_x)
follow_dist_avg_real = np.average(drone_x[0:-1] - future_hexa_x)

depth_real = np.abs(drone_x - hexa_x)  # Distance, so non-negative
depth_measured = np.copy(depth)
error_depth = depth_measured - depth_real

# Averages
# avg_error_est_x = np.average(np.array([*error_est_x[0:3], error_est_x[-1]])) # Remove outlier
avg_error_est_x = np.average(error_est_x) # Remove outlier
avg_error_est_y = np.average(error_est_y)
avg_error_depth = np.average(error_depth)

in_separator = "_"
out_separator = "_"
name_parts = exp_file.name.split(in_separator)[0:-1]  # strip last element with extension
name_parts.append("stats.txt")
stats_file = Path(out_separator.join(name_parts))
with open(stats_file, 'w') as f:
    f.write("Experiment Statistics\n")

    f.write("\nDrone Positional Error (Estimated - Real)\n")
    # f.write(f"X Error (m): {error_est_x}\n")
    # f.write(f"Y Error (m): {error_est_y}\n")
    f.write(f"Average Estimated Drone X Error (m): {avg_error_est_x}\n")
    f.write(f"Average Estimated Drone Y Error (m): {avg_error_est_y}\n")

    f.write("\nDepth Stats\n")
    # f.write(f"Depth Error (Measured - Real): {error_depth}\n")
    f.write(f"Average Measured Depth Error (m): {avg_error_depth}\n")

    f.write("\nFollow Distance Stats (m)\n")
    f.write(f"Avg Follow Distance (from estimated UAV pos): {follow_dist_avg_est}\n")
    f.write(f"Avg Follow Distance (from real UAV pos): {follow_dist_avg_real}\n")
#%% Matplotlib plots

fig, ax = plt.subplots()

ax.set(title="Map Frame X Positions", xlabel="Time (s)", ylabel="X (m)")
ax.grid()
ax.plot(t, drone_x, '-o', color='k', label="CoDrone Ground Truth")
ax.plot(t, est_x, '-o', color='r', label="Estimated CoDrone Position")
ax.plot(t, hexa_x, '-o', color='b', label="JetHexa")
ax.plot(future_hexa_t, future_hexa_x, '-o', color='g', label="JetHexa (Time Shifted)")
plt.legend()


y_fig, y_ax = plt.subplots()

y_ax.set(title="Map Frame Y Positions", xlabel="Time (s)", ylabel="Y (m)")
y_ax.grid()
y_ax.plot(t, drone_y, '-o', color='k', label="CoDrone Ground Truth")
y_ax.plot(t, est_y, '-o', color='r', label="Estimated CoDrone Position")
y_ax.plot(t, hexa_y, '-o', color='b', label="JetHexa")
y_ax.plot(future_hexa_t, future_hexa_y, '-o', color='g', label="JetHexa (Time Shifted)")
plt.legend()
plt.show()

