import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

base_file_path = os.path.abspath(os.path.join(os.curdir, '..', '..', '..')) # should point to the level above the src directory
data_path = os.path.join(base_file_path, 'data', 'IntercityFlow_LA')

city_data = pickle.load(open(os.path.join(data_path, 'data_processing_outputs', 'city_data.p'), 'rb'))
adjacency_matrix = np.load(os.path.join(data_path, 'data_processing_outputs', 'adjacency_matrix.npy'))

city_list = list(city_data.keys())
num_cities = len(city_list)

# Entity indexes
# 0 - groceries
# 1 - fitness
# 2 - pharmacy
# 3 - physician
# 4 - hotel
# 5 - religion
# 6 - restaurant
entity_list = ['grocery_demand_dest', 
                'fitness_demand_dest', 
                'pharmacy_demand_dest', 
                'physician_demand_dest', 
                'hotel_demand_dest',
                'restaurant_demand_dest']
num_entities = len(entity_list)

edge_weights = np.zeros((num_cities, num_cities, num_entities), dtype=np.float)

for city_ind in range(num_cities):
    for entity_ind in range(num_entities):
        city_name = city_list[city_ind]
        entity_name = entity_list[entity_ind]
        edge_weights[city_ind, :, entity_ind] = np.array(city_data[city_name][entity_name])

# Prune edge weights corresponding to missing edges in pre-calculated adjacency matrix
for entity_ind in range(num_entities):
    edge_weights[:, :, entity_ind] = np.multiply(adjacency_matrix, edge_weights[:, :, entity_ind]) # Use the adjacency matrix as a mask

for city_ind in range(num_cities):
    for entity_ind in range(num_entities):
        # weights are only considered to determine likelihood of moving to ANOTHER region. 
        # Should not take self-loops into account.
        edge_weights[city_ind, city_ind, entity_ind] = 0.0

        #normalize remaining edges to sum to 1
        if not (np.sum(edge_weights[city_ind, :, entity_ind]) == 0.0):
            edge_weights[city_ind, :, entity_ind] = edge_weights[city_ind, :, entity_ind] / np.sum(edge_weights[city_ind, :, entity_ind])
        
        # If no edge weight data exists, set weights to be uniform over the outgoing edges in the adjacency matrix
        else:
            num_outgoing_edges = np.sum(adjacency_matrix[city_ind, :])
            if adjacency_matrix[city_ind, city_ind] == 1:
                num_outgoing_edges = num_outgoing_edges - 1
            for adj_ind in range(num_cities):
                mask_val = adjacency_matrix[city_ind, adj_ind]
                if mask_val == 1.0 and not (adj_ind == city_ind):
                    edge_weights[city_ind, adj_ind, entity_ind] = 1 / num_outgoing_edges
                else:
                    edge_weights[city_ind, adj_ind, entity_ind] = 0.0

for city_ind in range(num_cities):
    for entity_ind in range(num_entities):
        if np.sum(edge_weights[city_ind, :, entity_ind]) - 1.0 <= 1e-8:
            pass
        else:
            print('city name: {}, entity name: {}, sum of edge weights: {}'.format(city_list[city_ind], 
                                                                        entity_list[entity_ind], 
                                                                        np.sum(edge_weights[city_ind, :, entity_ind])))

np.save(os.path.join(data_path, 'data_processing_outputs', 'edge_weights.npy'), edge_weights)

# entity = 6

# # Visualize the resulting adjacency matrix
# G = nx.DiGraph()
# G_fully_connected = nx.DiGraph()
# for i in range(num_cities):
#     G.add_node(city_list[i])
#     G_fully_connected.add_node(city_list[i])
#     for j in range(num_cities):
#         G_fully_connected.add_edge(city_list[i], city_list[j])
#         if edge_weights[i, j, entity] >= 1e-4:
#             G.add_edge(city_list[i], city_list[j])

# pos = dict()
# for i in range(num_cities):
#     city = city_list[i]
#     x_loc = city_data[city]['x_loc']
#     y_loc = city_data[city]['y_loc']
#     pos[city] = np.array([x_loc, y_loc])

# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos)
# # nx.draw_networkx_edges(G_fully_connected, pos, edge_color='red')
# nx.draw_networkx_edges(G, pos, edge_color='black', edge_width=15)

# plt.show()
