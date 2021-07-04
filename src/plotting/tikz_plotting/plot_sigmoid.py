# %%
import numpy as np
import sys
sys.path.append('../..')
from utils.tester import Tester
import pickle
import os
import matplotlib.pyplot as plt
import math

import tikzplotlib

# %%

city_name = 'Phoenix'
save_file_name = '2021-04-23_14-02-29'

fontsize = 20
legend_fontsize = 20

base_directory = os.getcwd()
base_directory = base_directory[0:base_directory.find('src')+3]

file_path = os.path.join(base_directory, 'optimization', 'save', save_file_name)
with open(file_path,'rb') as f:
    tester = pickle.load(f)

num_city = tester.params['m']
num_time = tester.params['n']
num_entity = tester.params['num_entity']
phi_val = tester.results['phi_best']
scale_frac = tester.params['scale_frac']

# %%

# Compute average lockdown over time
average_lockdown_array = np.zeros((num_city, num_entity))
L = np.array(tester.results['L_best'])
for city_ind in range(num_city):
    for entity_ind in range(num_entity):
        average_lockdown_array[city_ind, entity_ind] = 1 - np.average(L[:, city_ind, entity_ind])

populations = np.array(tester.problem_data['Ntot']) * scale_frac
demand = np.array(tester.problem_data['demand']) * scale_frac

# %%

entity_names = ['Grocery Stores', 'Fitness Centers', 'Pharmacies', 'Physicians', 'Hotels', 'Restaurants']
color_list = ['blue', 'red', 'green', 'magenta', 'black', 'orange']

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
for entity_ind in range(num_entity):
    ax.plot(demand[:, entity_ind], average_lockdown_array[:, entity_ind], linestyle='none', marker='d', color=color_list[entity_ind], markersize=15, label=entity_names[entity_ind])
ax.tick_params(axis='both', labelsize=fontsize)
ax.set_xscale('log')
ax.grid()
ax.legend(fontsize=legend_fontsize)

ax.set_ylabel(r'Average Lockdown Rate', fontsize=fontsize)
ax.set_xlabel(r'Entity Demand Rate [$\frac{people}{timestep}$]', fontsize=fontsize)

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_demand_lockdown_vs_demand.tex')
tikzplotlib.save(filename)
# %%

# %%
