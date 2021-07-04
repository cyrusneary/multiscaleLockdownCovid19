#%%
import numpy as np
import sys
sys.path.append('..')
from utils.tester import Tester
import pickle
import os
import matplotlib.pyplot as plt
import math
import networkx as nx
import random

city_name = 'Phoenix'
save_file_name = '2021-04-23_14-02-29'
seed = 45

# city_name = 'Seattle'
# save_file_name =  '2021-03-21_23-18-05'
# seed = 10

# city_name = 'Dallas'
# save_file_name =  '2021-04-09_21-11-28'

fontsize = 20
legend_fontsize = 20

# %%
# Load results data
base_directory = os.getcwd()
base_directory = base_directory[0:base_directory.find('src')+3]

file_path = os.path.join(base_directory, 'optimization', 'save', save_file_name)
with open(file_path,'rb') as f:
    tester = pickle.load(f)

if city_name == 'Phoenix':
    data_folder_name = 'Phoenix'
if city_name == 'Seattle':
    data_folder_name = 'IntercityFlow_Seattle'
if city_name == 'Dallas':
    data_folder_name = 'Intercity_Dallas'

# Load city data
city_data_file_path = os.path.join(base_directory, '..', 'data', data_folder_name, 'data_processing_outputs', 'city_data.p')
with open(city_data_file_path,'rb') as f:
    city_data = pickle.load(f)
city_list = list(city_data.keys())
num_cities = len(city_list)

num_city = tester.params['m']
num_time = tester.params['n']
num_entity = tester.params['num_entity']
phi_val = np.array(tester.results['phi_best'])
scale_frac = tester.params['scale_frac']

phi_average = np.zeros((num_city, num_city), dtype=np.float)
for i in range(num_city):
    for j in range(num_city):
        if not (i == j) and np.average(phi_val[:,i,j]) > 0.0:
            phi_average[i,j] = np.average(phi_val[:, i, j])

for city_ind in range(num_city):
    phi_average[city_ind,:] = phi_average[city_ind,:] * tester.problem_data['Ntot'][city_ind] * scale_frac

phi_average[:,:] = np.log(phi_average[:,:]+1e-3)

max_val = np.max(phi_average[:,:])
phi_average = phi_average / max_val
# print(phi_average)

# %%
edge_weight_list = []

# Visualize the resulting adjacency matrix
G = nx.DiGraph()
for i in range(num_cities):
    G.add_node(city_list[i])
    for j in range(num_cities):
        if phi_average[i,j] > 0.0:
            G.add_edge(city_list[i], city_list[j], weight=phi_average[i,j])
            edge_weight_list.append(phi_average[i,j])

if city_name == 'Dallas':
    city_data['Johnson']['y_loc'] = 32.385655
    city_data['Johnson']['x_loc'] = -97.335191
    city_data['Ellis']['y_loc'] = 32.362181
    city_data['Ellis']['x_loc'] = -96.803901
    city_data['Kaufman']['y_loc'] = 32.613997
    city_data['Kaufman']['x_loc'] = -96.283543
    city_data['Parker']['y_loc'] = 32.783855
    city_data['Parker']['x_loc'] = -97.802077
    city_data['Rockwall']['y_loc'] = 32.900920
    city_data['Rockwall']['x_loc'] = -96.404271
    city_data['Collin']['y_loc'] = 33.20671
    city_data['Collin']['x_loc'] = -96.587485
    city_data['Denton']['y_loc'] = 33.199884
    city_data['Denton']['x_loc'] = -97.089478
    city_data['Wise']['y_loc'] = 33.219515
    city_data['Wise']['x_loc'] = -97.647529
    city_data['Tarrant']['y_loc'] = 32.770195
    city_data['Tarrant']['x_loc'] = -97.264026
    city_data['Dallas']['y_loc'] = 32.77
    city_data['Dallas']['x_loc'] = -96.79

pos = dict()
for i in range(num_cities):
    city = city_list[i]
    x_loc = city_data[city]['x_loc']
    y_loc = city_data[city]['y_loc']
    pos[city] = np.array([x_loc, y_loc])

edge_width_list = np.array(edge_weight_list)
edge_width_list = np.exp(edge_width_list)
edge_width_list = edge_width_list / np.max(edge_width_list)
edge_width_list = edge_width_list * 5

options = {
    "node_color": "#A0CBE2",
    "edge_color": edge_weight_list,
    "node_size": tester.problem_data['Ntot'],
    "width": edge_width_list,
    "edge_cmap": plt.cm.Blues,
    "with_labels": False,
    "edge_vmin": 0.0,
    # "edge_vmax": 100.0
}

print(phi_average[1,:])

print(edge_weight_list)

random.seed(seed)
np.random.seed(seed=seed)
pos = nx.spring_layout(G)
# pos = nx.spectral_layout(G)

#print(city_data['Dallas']['population'])

# %%

plt.figure(figsize=(20,10))
# nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G_fully_connected, pos, edge_color='red')
nx.draw(G, pos, **options)

save_location = os.path.join(base_directory, 'plotting', city_name, 'saved_plots')
filename = os.path.join(save_location, '{}scale_cost_by_pop_phi_graph.png'.format(save_file_name))
plt.savefig(filename, bbox_inches='tight')
plt.show()
#plt.title('Adjacency Matrix with Scaled Demand Threshold {},\n Total Number of Edges: {}'.format(0.02, np.sum(adj_mat)), fontsize=15)

# %%
