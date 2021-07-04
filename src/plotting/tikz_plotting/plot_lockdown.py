
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

city_name = 'Phoenix'
# save_file_name = '2021-04-09_21-26-07'
save_file_name = '2021-04-23_14-02-29' # 

# city_name = 'Seattle'
# save_file_name =  '2021-03-21_23-18-05'

# city_name = 'Dallas'
# save_file_name =  '2021-04-09_20-36-29'

base_directory = os.getcwd()
base_directory = base_directory[0 : base_directory.find('src') + 3]

file_path = os.path.join(base_directory, 'optimization', 'save', save_file_name)
with open(file_path,'rb') as f:
    tester = pickle.load(f)

if city_name == 'Phoenix':
    data_folder_name = 'Phoenix'
if city_name == 'Seattle':
    data_folder_name = 'IntercityFlow_Seattle'
if city_name == 'Dallas':
    data_folder_name = 'Intercity_Dallas'

city_data_file_path = os.path.join(base_directory, '..', 'data', data_folder_name, 'data_processing_outputs', 'city_data.p')
with open(city_data_file_path,'rb') as f:
    city_data = pickle.load(f)

# %%

num_city = tester.params['m']
num_time = tester.params['n']
num_entity = tester.params['num_entity']
phi_val = tester.results['phi_best']
scale_frac = tester.params['scale_frac']

t = np.arange(0, num_time - 1)

# shape: (time, city_index, entity_index)
L_best = np.array(tester.results['L_best'])

# 0 - groceries
# 1 - fitness
# 2 - pharmacy
# 3 - physician
# 4 - hotel
# 5 - restaurant

entity_names = ['Grocery Stores', 'Fitness Centers', 'Pharmacies', 'Physicians', 'Hotels', 'Restaurants']

if city_name == 'Phoenix':
    cities_to_plot = ['Chandler', 'Phoenix', 'Cave Creek', 'Mesa', 'Tempe']
if city_name == 'Seattle':
    cities_to_plot = ['Seattle', 'Tacoma', 'Kent', 'Arlington', 'Everett']

if city_name == 'Dallas':
    cities_to_plot = ['Tarrant', 'Dallas', 'Collin', 'Ellis', 'Wise']


# cities_to_plot = list(city_data.keys())

color_list = ['blue', 'red', 'green', 'magenta', 'black', 'yellow']
cities_to_plot_indeces = []
for city in cities_to_plot:
    cities_to_plot_indeces.append(city_data[city]['index'])



# %%

fig = plt.figure()

for entity_ind in range(num_entity):
    ax = fig.add_subplot(2, 3, entity_ind + 1)
    for i in range(len(cities_to_plot)):
        city_ind = cities_to_plot_indeces[i]
        name = cities_to_plot[i]
        ax.plot(t, 1 - L_best[0:-1, city_ind, entity_ind], label=name, color=color_list[i])
    if entity_ind >= 3:
        ax.set_xlabel('Time [days]')
    ax.set_ylabel('Lockdown Rate')
    ax.set_title(entity_names[entity_ind])
    ax.set_xlim(0, num_time - 1)
    ax.set_ylim(0, 1)
    
    if entity_ind in [1, 2, 4, 5]:
        ax.set_yticks([])
    if entity_ind in [0, 1, 2]:
        ax.set_xticks([])

    if entity_ind == 4:
        plt.legend()

save_location = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(save_location, 'scale_cost_by_pop_lockdown.tex')
tikzplotlib.save(filename)


# %%

# %%
