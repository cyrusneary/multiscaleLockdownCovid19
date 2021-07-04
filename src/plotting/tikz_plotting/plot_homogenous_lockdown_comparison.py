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
# data.append({'save_file_name': '2021-03-20_03-48-37', 'description': 'Fully Targeted Lockdown'})
# data.append({'save_file_name': '2021-03-20_03-20-17', 'description': 'Lockdown fixed for entities'})
# data.append({'save_file_name': '2021-03-20_03-20-13', 'description': 'Lockdown fixed for cities'})
# data.append({'save_file_name': '2021-03-20_03-23-06', 'description': 'Lockdown fixed for all'})

# data.append({'save_file_name': '2021-04-09_21-26-07', 'description': 'Fully Targeted Lockdown'})
# data.append({'save_file_name': '2021-04-09_21-27-44', 'description': 'Lockdown fixed for entities'})
# data.append({'save_file_name': '2021-04-09_21-18-40', 'description': 'Lockdown fixed for cities'})
# data.append({'save_file_name': '2021-04-09_21-19-03', 'description': 'Lockdown fixed for all'})

data.append({'save_file_name': '2021-04-23_14-02-29', 'description': 'Fully Targeted Lockdown'})
data.append({'save_file_name': '2021-04-25_13-24-31', 'description': 'Lockdown fixed for cities'})
data.append({'save_file_name': '2021-04-25_13-32-06', 'description': 'Lockdown fixed for entities'})
data.append({'save_file_name': '2021-04-25_13-13-27', 'description': 'Lockdown fixed for all'})

# city_name = 'Seattle'
# save_file_name =  '2021-03-21_23-18-05'

base_directory = os.getcwd()
base_directory = base_directory[0:base_directory.find('src')+3]

if city_name == 'Phoenix':
    data_folder_name = 'Phoenix'
if city_name == 'Seattle':
    data_folder_name = 'IntercityFlow_Seattle'

# Load city data
city_data_file_path = os.path.join(base_directory, '..', 'data', data_folder_name, 'data_processing_outputs', 'city_data.p')
with open(city_data_file_path,'rb') as f:
    city_data = pickle.load(f)
city_list = list(city_data.keys())

# %%

total_population = 0
for city in city_data.keys():
    total_population = total_population + city_data[city]['population']
total_population = total_population[0]

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
    data[ind]['peak_infections'] = np.max(data[ind]['I']) * 100000 / total_population
    data[ind]['num_deaths'] = data[ind]['D'][-1] * 100000 / total_population
    # data[ind]['peak_infections'] = 100 * np.max(data[ind]['I']) / total_population
    # data[ind]['num_deaths'] = 100 * data[ind]['D'][-1] / total_population
    peak_infections_list.append(data[ind]['peak_infections'])
    num_deaths_list.append(data[ind]['num_deaths'])
    average_lockdown_list.append(1 - np.average(data[ind]['tester'].results['L_best'][0:98]))

# %%

# Plot the results
x = np.arange(len(data))

cmap = matplotlib.cm.get_cmap('Oranges')
norm = matplotlib.colors.Normalize(vmin=np.min(num_deaths_list), vmax=np.max(num_deaths_list))
color_list = []
for i in range(len(data)):
    color_list.append(cmap(norm(data[i]['num_deaths'])))

width = 0.5

# %%

fig = plt.figure()

### PLOT PEAK INFECTIONS COMPARISON
ax1 = fig.add_subplot(111)
cmap = matplotlib.cm.get_cmap('Oranges')
norm = matplotlib.colors.Normalize(vmin=np.min(num_deaths_list), vmax=np.max(num_deaths_list))

labels = []
for i in range(len(data)):
    val = data[i]['peak_infections']
    ax1.bar(i, val, width, edgecolor='black', facecolor=color_list[i], label=data[i]['description'])

labels = [
    'Heterogeneous \n Lockdown \n Strategies', 
    'Activity Site \n Lockdown \n Strategies', 
    'Regional \n Lockdown \n Strategies', 
    'MSA \n Lockdown \n Strategies'
    ]

# ax1.set_title('Peak Infections', fontsize=fontsize)
ax1.set_ylabel('Peak Infections per 100,000 of Population')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_homogenous_lockdown_infections_comparison.tex')
tikzplotlib.save(filename)

# %%

fig = plt.figure()

### PLOT DEATHS COMPARISON
ax2 = fig.add_subplot(111)
cmap = matplotlib.cm.get_cmap('Oranges')
norm = matplotlib.colors.Normalize(vmin=np.min(num_deaths_list), vmax=np.max(num_deaths_list))

labels = []
for i in range(len(data)):
    val = data[i]['num_deaths']
    ax2.bar(i, val, width, edgecolor='black', facecolor=color_list[i], label=data[i]['description'])

labels = [
    'Heterogeneous \n Lockdown \n Strategies', 
    'Activity Site \n Lockdown \n Strategies', 
    'Regional \n Lockdown \n Strategies', 
    'MSA \n Lockdown \n Strategies'
    ]

# ax1.set_title('Peak Infections', fontsize=fontsize)
ax2.set_ylabel('Deaths per 100,000 People')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.tick_params(axis='both')


save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_homogenous_lockdown_deaths_comparison.tex')
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
    'Heterogeneous \n Lockdown \n Strategies', 
    'Activity Site \n Lockdown \n Strategies', 
    'Regional \n Lockdown \n Strategies', 
    'MSA \n Lockdown \n Strategies'
    ]

ax3.set_ylabel('Average Lockdown')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.tick_params(axis='both')

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_homogenous_lockdown_avg_lockdown_comparison.tex')

tikzplotlib.save(filename)

# %%

# %%
