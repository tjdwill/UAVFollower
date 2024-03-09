# -*- coding: utf-8 -*-
"""
Spyder Editor

This file generates the position and error plots, the following distance plot, and relevant statistics.
The data should be formatted via CSV and should already be time-synchronized as much as possible.
This is more exploratory programming in nature, so it helps to use Spyder to view the variable values
and interact/experiment with the data via the iPython command line.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
import numpy as np
from os import chdir
from pathlib import Path


# Initial Config
mpl.rcParams['figure.dpi'] = 300  # Hi-res figures
plt.style.use('tableau-colorblind10')
plt.style.use(['science', 'ieee'])
plt.rc('legend', fontsize=6)

FOLLOW_DIST = 0.3  # meters

# Pathing: Load files
chdir(Path("C:/Users/tjdwill/Desktop/20240308_0145_uav_follower_analysis/bags/following_experiments"))

combined_exp_file = Path("exp03/csv/synced_uav_follow_exp_2024-03-05-19-00-55_combined.csv")
codrone_exp_file = Path("exp03/csv/focused_uav_pos_2024-03-05_19-03-45.csv")
jethexa_exp_file = Path("exp03/csv/focused_uav_jethexa_exp_2024-03-05-19-00-55_jethexa_pos.csv")

# Read the data into the variables

## Combined Data
combined_csv = pd.read_csv(combined_exp_file)
columns = combined_csv.columns.values.tolist()

x_cov_offset = 0.247
combined_t, drone_x, drone_y, est_x, est_y, hexa_x, hexa_y, depth = columns
combined_t = combined_csv[combined_t].values

drone_x = combined_csv[drone_x].values
drone_y = combined_csv[drone_y].values

est_x = combined_csv[est_x].values
est_y = combined_csv[est_y].values

hexa_x = combined_csv[hexa_x].values
hexa_y = combined_csv[hexa_y].values

depth = combined_csv[depth].values

## Drone data
codrone_csv = pd.read_csv(codrone_exp_file)
### Get column names
codrone_cols = codrone_csv.columns.values.tolist()
codrone_data = []
for col_name in codrone_cols:
    codrone_data.append(codrone_csv[col_name].values)
codrone_t, codrone_x, codrone_y = codrone_data

## FULL JetHexa Data
jethexa_csv = pd.read_csv(jethexa_exp_file)
### Get column names
jethexa_cols = jethexa_csv.columns.values.tolist()
jethexa_data = []
for col_name in jethexa_cols:
    jethexa_data.append(jethexa_csv[col_name].values)
jethexa_t, jethexa_x, jethexa_y = jethexa_data[:3]

## Adjust time data
TIME_OFFSET = np.min([*combined_t, *codrone_t, *jethexa_t])
time_data = [combined_t, jethexa_t, codrone_t]
for arr in time_data:
    arr -= TIME_OFFSET

## Define Time-shifted JH
future_hexa_x = hexa_x[1:]
future_hexa_y = hexa_y[1:]
future_hexa_t = np.copy(combined_t[0:-1])

# %% Report Statistics

## Get rid of last outlier point
est_x = est_x[:-1]
est_y = est_y[:-1]
drone_x = drone_x[:-1]
drone_y = drone_y[:-1]
hexa_x = hexa_x[:-1]
hexa_y = hexa_y[:-1]
depth = depth[:-1]
combined_t = combined_t[:-1]

error_est_x = est_x - drone_x
error_est_y = est_y - drone_y
follow_dist_est_x = est_x - future_hexa_x
follow_dist_real_x = drone_x - future_hexa_x
follow_dist_est_y = est_y - future_hexa_y
follow_dist_real_y = drone_y - future_hexa_y
follow_dist_est = np.sqrt(follow_dist_est_x**2 + follow_dist_est_y**2)
follow_dist_real = np.sqrt(follow_dist_real_x**2 + follow_dist_real_y**2)
depth_real = np.abs(drone_x - hexa_x)  # Distance, so non-negative
depth_measured = np.copy(depth)
error_depth = depth_measured - depth_real

# Means
decimal_place = 3

mean_error_est_x = np.average(np.abs(error_est_x)).round(decimal_place)
mean_error_est_y = np.average(np.abs(error_est_y)).round(decimal_place)

mean_follow_err_x = np.average(np.abs(follow_dist_est_x - follow_dist_real_x)).round(decimal_place)
mean_follow_err = np.average(np.abs(follow_dist_est - follow_dist_real)).round(decimal_place)

# Medians
median_error_est_x = np.median(np.abs(error_est_x)).round(decimal_place)
median_error_est_y = np.median(np.abs(error_est_y)).round(decimal_place)

median_follow_err_x = np.median(np.abs(FOLLOW_DIST - follow_dist_real_x)).round(decimal_place)
median_follow_err = np.median(np.abs(FOLLOW_DIST - follow_dist_real)).round(decimal_place)


#%% Write Results to File
in_separator = "_"
out_separator = "_"
name_parts = combined_exp_file.name.split(in_separator)[0:-1]  # strip last element with extension
name_parts.append("stats.txt")
stats_file = Path(out_separator.join(name_parts))
with open(stats_file, 'w') as f:
    f.write("Experiment Statistics (Mean, Median)\n")

    f.write("\n::Drone Positional Error (Estimated - Real)::\n")

    f.write(f"Estimated Drone X Error (m): {mean_error_est_x}, {median_error_est_x}\n")
    f.write(f"Estimated Drone Y Error (m): {mean_error_est_y}, {median_error_est_y}\n")

    f.write("\n::Follow Distance Error (m)::\n")
    f.write(f"X Follow Distance: {mean_follow_err_x}, {median_follow_err_x}\n")
    f.write(f"Radial Follow Distance: {mean_follow_err}, {median_follow_err}\n")

#%% Matplotlib plots

codrone_clr = 'k'
jethexa_clr = 'b'
shift_jh_clr = 'g'
est_clr = 'r'
marker_size=2
cap_size = 1.25*marker_size
max_time_tick = 300 # exclusive
tick_step = 30
fig, ax = plt.subplots()
xticks = np.arange(0, max_time_tick, tick_step)
zorder = 5
scatter_sz = 5

## Position Estimation Error
ax.set(
    title="Map Frame X Positions",
    xlabel="Time (s)", ylabel="X (m)",
    xticks=xticks
)
ax.grid()
ax.plot(
    codrone_t, codrone_x,
    marker='.', markersize=marker_size,
    color=codrone_clr, label="CoDrone Ground Truth"
)
ax.scatter(
    combined_t, est_x,
    color=est_clr, label="Estimated CoDrone Position",
    zorder=zorder, s=marker_size
)
ax.plot(
    jethexa_t, jethexa_x,
    marker='.', markersize=marker_size,
    color=jethexa_clr, label="JetHexa"
)
plt.legend()


y_fig, y_ax = plt.subplots()

y_ax.set(
    title="Map Frame Y Positions",
    xlabel="Time (s)", ylabel="Y (m)", xticks=xticks
)
y_ax.grid()
y_ax.plot(
    codrone_t, codrone_y,
    marker='.', markersize=marker_size,
    color=codrone_clr, label="CoDrone Ground Truth"
)
y_ax.scatter(
    combined_t, est_y,
    color=est_clr, label="Estimated CoDrone Position",
    zorder=zorder, s=marker_size
)
y_ax.plot(
    jethexa_t, jethexa_y,
    marker='.', markersize=marker_size,
    color=jethexa_clr, label="JetHexa"
)

plt.legend()
plt.show()

## Position Estimation Error (Bar Graphs)
abs_error_est_x = np.abs(error_est_x)
abs_error_est_y = np.abs(error_est_y)
bar_width = 1.2
fig, ax = plt.subplots()
ax.set(
    title="X Position Estimation Error",
    xlabel="Time (s)", ylabel="Error (m)",
    xticks=xticks
)
ax.grid()
ax.bar(
       combined_t, abs_error_est_x, width=bar_width,
       color='r', zorder=5
)

fig, ax = plt.subplots()
ax.set(
    title="Y Position Estimation Error",
    xlabel="Time (s)", ylabel="Error (m)",
    xticks=xticks
)
ax.grid()
ax.bar(
       combined_t, abs_error_est_y, width=bar_width,
       color='r', zorder=5
)

# %% Follow Distance Plot

"""
The CoDrone and JetHexa data are not aligned in terms of the number of elements, meaning the distance cannot be calculated for every point in both sets of data.
Therefore, I need to sample the larger set of data for points within the time range of the smaller set.

The codrone has more data.
"""

num_samples = jethexa_t.shape[0]

"""
Via the Variable Explorer in Spyder, I can see that the first time element in CoDrone data that corresponds to the JetHexa data is at element 140.
The JetHexa Data is further apart in time (avg. of about .4 seconds). The CoDrone data is about .15 seconds apart, so I take every three points.
"""
codrone_cat_orig = np.concatenate((codrone_t.reshape(-1, 1), codrone_x.reshape(-1, 1), codrone_y.reshape(-1, 1)), axis=1)

codrone_cat = codrone_cat_orig[140::3]


"""
From here, the codrone data has 340 points to Jethexa's 372 for this data iteration, and they are much closer in time spans.
Now, I sample JetHexa's data in order to get 340 points. Naturally, this will change for different runs.
"""
jethexa_cat = np.concatenate((jethexa_t.reshape(-1, 1), jethexa_x.reshape(-1, 1), jethexa_y.reshape(-1, 1)), axis=1)

# Use Numpy sampling method
rng = np.random.default_rng()
jethexa_sampled = rng.choice(
    jethexa_cat, size=codrone_cat.shape[0],
    replace=False, axis=0, shuffle=False
)

jethexa_sampled = np.sort(jethexa_sampled, axis=0)
time_errors = np.abs(jethexa_sampled[:,0] - codrone_cat[:,0])

## Sample until we get a good batch; Don't want significant time error (use a powerful computer).
TIME_ERR_THRESH = 1
ATTEMPT_OUT = 10000000
count = 0
while np.max(time_errors) > TIME_ERR_THRESH and count < ATTEMPT_OUT:
    jethexa_sampled = rng.choice(
        jethexa_cat, size=codrone_cat.shape[0],
        replace=False, axis=0, shuffle=False
    )
    jethexa_sampled = np.sort(jethexa_sampled, axis=0)
    time_errors = np.abs(jethexa_sampled[:,0] - codrone_cat[:,0])
    count += 1
    #print(np.max(time_errors))
else:
    if count >= ATTEMPT_OUT:
        raise ValueError(f"Could not get a sample with less than {TIME_FOR_THRESH} time error.\n")

print(f"Found a good sample on the {count}th try.\n")
median_time_error = np.median(time_errors)
mean_time_error = np.mean(time_errors)
distances = jethexa_sampled[:,1:] - codrone_cat[:,1:]
distances = distances ** 2
follow_dist = np.sum(distances, axis=1)
median_follow_dist = np.median(follow_dist)
mean_follow_dist = np.mean(follow_dist)

# %%% Plot the follow distances
fig, ax = plt.subplots()
ax.set(
    title="Follow Distance",
    xlabel="Time (s)", ylabel="Distance (m)",
    xticks=xticks
)
ax.grid()
ax.axhline(FOLLOW_DIST, color='k', linestyle="--", label="Target Distance")
ax.plot(
        jethexa_sampled[:,0], follow_dist,
        'g-o', markersize=marker_size
)
plt.legend()
