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

# data.append({'save_file_name': '2021-04-09_21-26-07', 'description': '3 edges'})
# data.append({'save_file_name': '2021-04-09_23-33-48', 'description': '4 edges'})
# data.append({'save_file_name': '2021-04-10_00-51-46', 'description': '5 edges'})

data.append({'save_file_name': '2021-04-23_14-02-29', 'description': '3 edges'})
data.append({'save_file_name': '2021-04-25_14-21-43', 'description': '4 edges'})
data.append({'save_file_name': '2021-04-25_18-01-31', 'description': '5 edges'})

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
num_edges_list = []
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
    data[ind]['num_edges'] = int(np.sum(data[ind]['tester'].problem_data['adj_mat']))

# %%

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
    '3', 
    '4', 
    '5', 
    ]

# ax1.set_title('Peak Infections', fontsize=fontsize)
ax1.set_ylabel('Infections per 100,000')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_different_edges_infections_comparison.tex')

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
    '3', 
    '4', 
    '5', 
    ]

ax2.set_ylabel('Deaths per 100,000')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_different_edges_deaths_comparison.tex')

tikzplotlib.save(filename)
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
    '3', 
    '4', 
    '5', 
    ]

ax3.set_ylabel('Average Lockdown')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_different_edges_lockdown_comparison.tex')

tikzplotlib.save(filename)
# %%
