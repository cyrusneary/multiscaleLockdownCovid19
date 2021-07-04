import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

base_file_path = os.path.abspath(os.path.join(os.curdir, '..', '..', '..')) # should point to the level above the src directory
data_path = os.path.join(base_file_path, 'data', 'Intercity_Dallas')

county_data = pickle.load(open(os.path.join(data_path, 'data_processing_outputs', 'county_data.p'), 'rb'))
adjacency_matrix = np.load(os.path.join(data_path, 'data_processing_outputs', 'adjacency_matrix_5.npy'))

county_list = list(county_data.keys())
num_counties = len(county_list)

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

edge_weights = np.zeros((num_counties, num_counties, num_entities), dtype=np.float)

for county_ind in range(num_counties):
    for entity_ind in range(num_entities):
        city_name = county_list[county_ind]
        entity_name = entity_list[entity_ind]
        edge_weights[county_ind, :, entity_ind] = np.array(county_data[city_name][entity_name])

# Prune edge weights corresponding to missing edges in pre-calculated adjacency matrix
for entity_ind in range(num_entities):
    edge_weights[:, :, entity_ind] = np.multiply(adjacency_matrix, edge_weights[:, :, entity_ind]) # Use the adjacency matrix as a mask

for county_ind in range(num_counties):
    for entity_ind in range(num_entities):
        # weights are only considered to determine likelihood of moving to ANOTHER region. 
        # Should not take self-loops into account.
        edge_weights[county_ind, county_ind, entity_ind] = 0.0

        #normalize remaining edges to sum to 1
        if not (np.sum(edge_weights[county_ind, :, entity_ind]) == 0.0):
            edge_weights[county_ind, :, entity_ind] = edge_weights[county_ind, :, entity_ind] / np.sum(edge_weights[county_ind, :, entity_ind])
        
        # If no edge weight data exists, set weights to be uniform over the outgoing edges in the adjacency matrix
        else:
            num_outgoing_edges = np.sum(adjacency_matrix[county_ind, :])
            if adjacency_matrix[county_ind, county_ind] == 1:
                num_outgoing_edges = num_outgoing_edges - 1
            for adj_ind in range(num_counties):
                mask_val = adjacency_matrix[county_ind, adj_ind]
                if mask_val == 1.0 and not (adj_ind == county_ind):
                    edge_weights[county_ind, adj_ind, entity_ind] = 1 / num_outgoing_edges
                else:
                    edge_weights[county_ind, adj_ind, entity_ind] = 0.0

for county_ind in range(num_counties):
    for entity_ind in range(num_entities):
        if np.sum(edge_weights[county_ind, :, entity_ind]) - 1.0 <= 1e-8:
            pass
        else:
            print('city name: {}, entity name: {}, sum of edge weights: {}'.format(county_list[county_ind], 
                                                                        entity_list[entity_ind], 
                                                                        np.sum(edge_weights[county_ind, :, entity_ind])))

np.save(os.path.join(data_path, 'data_processing_outputs', 'edge_weights_5.npy'), edge_weights)

print(edge_weights)

# entity = 5

# # Visualize the resulting adjacency matrix
# G = nx.DiGraph()
# G_fully_connected = nx.DiGraph()
# for i in range(num_counties):
#     G.add_node(county_list[i])
#     G_fully_connected.add_node(county_list[i])
#     for j in range(num_counties):
#         G_fully_connected.add_edge(county_list[i], county_list[j])
#         if edge_weights[i, j, entity] >= 1e-4:
#             G.add_edge(county_list[i], county_list[j])

# pos = dict()
# for i in range(num_counties):
#     county = county_list[i]
#     # x_loc = county_data[city]['x_loc']
#     # y_loc = county_data[city]['y_loc']
#     # pos[city] = np.array([x_loc, y_loc])
#     pos[county] = np.array([np.random.normal(), np.random.normal()])

# nx.draw_networkx_nodes(G, pos)
# nx.draw_networkx_labels(G, pos)
# # nx.draw_networkx_edges(G_fully_connected, pos, edge_color='red')
# nx.draw_networkx_edges(G, pos, edge_color='black')

# plt.show()
