# %%
import numpy as np
import sys
sys.path.append('../..')
from utils.tester import Tester
import pickle
import os
import matplotlib
import matplotlib.pyplot as plt
import math

import tikzplotlib

city_name = 'Phoenix'

data = []
# data.append({'save_file_name': '2021-04-09_16-35-32', 'description': 'Regular cost'})
# data.append({'save_file_name': '2021-04-09_16-38-49', 'description': '2x cost'})
# data.append({'save_file_name': '2021-04-09_16-44-04', 'description': '4x cost'})
# data.append({'save_file_name': '2021-04-09_16-46-39', 'description': '8x cost'})
# data.append({'save_file_name': '2021-04-09_18-01-51', 'description': '20x cost'})

# data.append({'save_file_name': '2021-04-09_21-26-07', 'description': 'Regular cost'})
# data.append({'save_file_name': '2021-04-11_12-57-45', 'description': '2x cost'})
# data.append({'save_file_name': '2021-04-11_12-53-45', 'description': '4x cost'})
# data.append({'save_file_name': '2021-04-11_13-01-47', 'description': '8x cost'})
# data.append({'save_file_name': '2021-04-11_12-51-09', 'description': '20x cost'})
# param_vals = [0.01, 0.02, 0.04, 0.08, 0.2]

data.append({'save_file_name': '2021-04-24_12-35-23', 'description': '0.1x cost'})
data.append({'save_file_name': '2021-04-24_12-34-29', 'description': '0.2x cost'})
data.append({'save_file_name': '2021-04-24_12-20-26', 'description': '0.5x cost'})
data.append({'save_file_name': '2021-04-23_14-02-29', 'description': 'Regular cost'})
data.append({'save_file_name': '2021-04-23_15-02-03', 'description': '2x cost'})
data.append({'save_file_name': '2021-04-23_15-00-02', 'description': '4x cost'})
param_vals = np.array([0.1, 0.2, 0.5, 1, 2, 4]) * 1e-4

base_directory = os.getcwd()
base_directory = base_directory[0:base_directory.find('src')+3]

if city_name == 'Phoenix':
    data_folder_name = 'Phoenix'
if city_name == 'Seattle':
    data_folder_name = 'IntercityFlow_Seattle'
if city_name == 'Dallas':
    data_folder_name = 'Intercity_Dallas'

# Load county data
county_data_file_path = os.path.join(base_directory, '..', 'data', data_folder_name, 'data_processing_outputs', 'city_data.p')
with open(county_data_file_path,'rb') as f:
    county_data = pickle.load(f)
county_list = list(county_data.keys())

# %%

total_population = 0
for county in county_data.keys():
    total_population = total_population + county_data[county]['population']

tester_list = []
peak_infections_list = []
num_deaths_list = []
average_lockdown_list = []
for ind in range(len(data)):
    file_path = os.path.join(base_directory, 'optimization', 'save', data[ind]['save_file_name'])
    with open(file_path,'rb') as f:
        tester = pickle.load(f)
        data[ind]['tester'] = tester
    data[ind]['scale_frac'] = tester.params['scale_frac']
    data[ind]['I'] = np.sum(tester.results['I_best'] * data[ind]['scale_frac'], axis=1)
    data[ind]['D'] = np.sum(tester.results['D_best'] * data[ind]['scale_frac'], axis=1)
    # data[ind]['peak_infections'] = 100 * np.max(data[ind]['I']) / 100,000 total_population
    # data[ind]['num_deaths'] = 100 * data[ind]['D'][-1] / total_population
    data[ind]['peak_infections'] = np.max(data[ind]['I']) * 100000 / total_population
    data[ind]['num_deaths'] = data[ind]['D'][-1] * 100000 / total_population
    peak_infections_list.append(data[ind]['peak_infections'])
    num_deaths_list.append(data[ind]['num_deaths'])
    average_lockdown_list.append(1 - np.average(data[ind]['tester'].results['L_best'][0:98]))

# %%

fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.plot(param_vals, [data[i]['peak_infections'] for i in range(len(data))], marker='d')
ax1.set_xscale('log')
ax1.set_xlabel('Economic Impact Parameter')
ax1.set_ylabel('Infections per 100,000')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_different_cost_infections_comparison.tex')
tikzplotlib.save(filename)

# %%
fig = plt.figure()

ax2 = fig.add_subplot(111)

ax2.plot(param_vals, [data[i]['num_deaths'] for i in range(len(data))], marker='d')
ax2.set_xscale('log')
ax2.set_xlabel('Economic Impact Parameter')
ax2.set_ylabel('Deaths per 100,000')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_different_cost_deaths_comparison.tex')
tikzplotlib.save(filename)

# %%
fig = plt.figure()

ax3 = fig.add_subplot(111)

ax3.plot(param_vals, average_lockdown_list, marker='d')
ax3.set_xscale('log')
ax3.set_xlabel('Economic Impact Parameter')
ax3.set_ylabel('Average Lockdown Rate')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_different_cost_lockdown_comparison.tex')
tikzplotlib.save(filename)

# %%%%%%%%%% OLD BAR PLOTS 

# Plot the results
x = np.arange(len(data))

cmap = matplotlib.cm.get_cmap('Oranges')
norm = matplotlib.colors.Normalize(vmin=np.min(peak_infections_list), vmax=np.max(peak_infections_list))
color_list = []
for i in range(len(data)):
    color_list.append(cmap(norm(data[i]['peak_infections'])))

width = 0.5

# %%

fig = plt.figure()

### PLOT PEAK INFECTIONS COMPARISON
ax1 = fig.add_subplot(111)

labels = []
for i in range(len(data)):
    val = data[i]['peak_infections']
    ax1.bar(i, val, width, edgecolor='black', facecolor=color_list[i], label=data[i]['description'])

labels = [
    '0.01', 
    '0.02', 
    '0.04', 
    '0.08',
    '0.20'
    ]

# ax1.set_title('Peak Infections', fontsize=fontsize)
ax1.set_ylabel('Peak Infections per 100,000 People')
ax1.set_xlabel('Economic Impact Parameter')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'different_cost_infections_comparison.tex')
tikzplotlib.save(filename)

# %%

fig = plt.figure()

### PLOT DEATHS COMPARISON
ax2 = fig.add_subplot(111)
cmap = matplotlib.cm.get_cmap('Oranges')
norm = matplotlib.colors.Normalize(vmin=np.min(num_deaths_list), vmax=np.max(num_deaths_list))

color_list = []
for i in range(len(data)):
    color_list.append(cmap(norm(data[i]['num_deaths'])))

labels = []
for i in range(len(data)):
    val = data[i]['num_deaths']
    ax2.bar(i, val, width, edgecolor='black', facecolor=color_list[i], label=data[i]['description'])

labels = [
    '0.01', 
    '0.02', 
    '0.04', 
    '0.08',
    '0.20'
    ]

ax2.set_ylabel('Deaths per 100,000 People')
ax2.set_xlabel('Economic Impact Parameter')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'different_cost_deaths_comparison.tex')

tikzplotlib.save(filename)
# %%
# %%

fig = plt.figure()

### PLOT lockdown COMPARISON
ax3 = fig.add_subplot(111)
cmap = matplotlib.cm.get_cmap('Blues')
norm = matplotlib.colors.Normalize(vmin=np.min(average_lockdown_list), vmax=np.max(average_lockdown_list))

color_list = []
for i in range(len(data)):
    color_list.append(cmap(norm(average_lockdown_list[i])))

labels = []
for i in range(len(data)):
    val = average_lockdown_list[i]
    ax3.bar(i, val, width, edgecolor='black', facecolor=color_list[i], label=data[i]['description'])

labels = [
    '0.01', 
    '0.02', 
    '0.04', 
    '0.08',
    '0.20'
    ]

ax3.set_ylabel('Average Lockdown')
ax3.set_xlabel('Economic Impact Parameter')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'different_cost_lockdown_comparison.tex')

tikzplotlib.save(filename)
# %%
