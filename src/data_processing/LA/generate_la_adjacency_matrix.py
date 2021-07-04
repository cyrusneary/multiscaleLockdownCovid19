import numpy as np
import pickle
import os
import networkx as nx
import matplotlib.pyplot as plt

base_file_path = os.path.abspath(os.path.join(os.curdir, '..', '..', '..')) # should point to the level above the src directory
data_path = os.path.join(base_file_path, 'data', 'IntercityFlow_LA')

city_data = pickle.load(open(os.path.join(data_path, 'data_processing_outputs', 'city_data.p'), 'rb'))

city_list = list(city_data.keys())
num_cities = len(city_list)

print(city_list)

# Entity indexes
# 0 - groceries
# 1 - fitness
# 2 - pharmacy
# 3 - physician
# 4 - hotel
# 5 - restaurant

# Build a matrix aggregating demand between cities over the different entities.
aggregate_intercity_demand_mat = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        city = city_list[i]
        aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['grocery_demand_dest'][j]
        aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['fitness_demand_dest'][j]
        aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['pharmacy_demand_dest'][j]
        aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['physician_demand_dest'][j]
        aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['hotel_demand_dest'][j]
        # aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['religion_demand_dest'][j]
        aggregate_intercity_demand_mat[i,j] = aggregate_intercity_demand_mat[i,j] + city_data[city]['restaurant_demand_dest'][j]

num_edges_to_keep = 3
adjacency_matrix = np.zeros((num_cities, num_cities), dtype=np.float)
for i in range(num_cities):
    indeces_to_keep = np.argsort(-aggregate_intercity_demand_mat[i,:])[0:num_edges_to_keep]
    adjacency_matrix[i, indeces_to_keep] = 1.0
    adjacency_matrix[i,i] = 1.0

# # Scale the aggregated demand by the population in the region
# scaled_aggregate_intercity_demand_mat = np.array(aggregate_intercity_demand_mat)
# for i in range(num_cities):
#     scaled_aggregate_intercity_demand_mat[i,:] = scaled_aggregate_intercity_demand_mat[i,:] / city_data[city_list[i]]['population']
# adjacency_matrix_threshold = 0.02
# adjacency_matrix = np.array(scaled_aggregate_intercity_demand_mat > adjacency_matrix_threshold, dtype=float)

np.save(os.path.join(data_path, 'data_processing_outputs', 'adjacency_matrix.npy'), adjacency_matrix)

# Visualize the resulting adjacency matrix
G = nx.DiGraph()
G_fully_connected = nx.DiGraph()
for i in range(num_cities):
    G.add_node(city_list[i])
    G_fully_connected.add_node(city_list[i])
    for j in range(num_cities):
        G_fully_connected.add_edge(city_list[i], city_list[j])
        if adjacency_matrix[i,j] == 1.0:
            G.add_edge(city_list[i], city_list[j])

pos = dict()
for i in range(num_cities):
    city = city_list[i]
    #x_loc = city_data[city]['x_loc']
    #y_loc = city_data[city]['y_loc']
    #pos[city] = np.array([x_loc, y_loc])
    # adding random values for the locations of counties of Dallas
    pos[city] = np.array([np.random.normal(), np.random.normal()])
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edges(G_fully_connected, pos, edge_color='red')
nx.draw_networkx_edges(G, pos, edge_color='black', width=4)

plt.title('Total Number of Edges: {}'.format(np.sum(adjacency_matrix)), fontsize=15)
plt.show()
