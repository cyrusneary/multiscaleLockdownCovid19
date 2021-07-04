import sys, os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

base_file_path = os.path.abspath(os.path.join(os.curdir, '..', '..', '..')) # should point to the level above the src directory
data_path = os.path.join(base_file_path, 'data', 'Phoenix')

# (grocery_demand, fitness_demand, pharmacy_demand, physician_demand, hotel_demand, religion_demand, restaurant_demand)
# Entity indexes
# 0 - groceries
# 1 - fitness
# 2 - pharmacy
# 3 - physician
# 4 - hotel
# 5 - restaurant

# Data processing parameters
fitness_freq = 94/12 # visits per unique visitor per month
pharmacy_freq = 35/12 # visits per unique visitor per month
physician_freq = 1 # visits per unique visitor per month
hotel_freq = 1 # visits per unique visitor per month
# religion_freq = 25/12 # visits per unique visitor per month
grocery_freq = 2 # visits per unique visitor per month
restaurant_freq = 1 # Assume each restaurant-goer only visits a given restaurant once per month (if at all)

month_day_time_conversion = 1/30 # months/day

min_demand_val = 5

# Find the average x-y locations of the various cities
print('Processing GEOID data...')
geoid = pd.read_excel(os.path.join(data_path, 'GEOID_CityXY.xlsx'))
cities = list(geoid.City.unique())
city_data = dict()
city_xy = dict()
for city in cities:
    x_loc = np.average(geoid.X_blockgroup[geoid.City == city])
    y_loc = np.average(geoid.Y_blockgroup[geoid.City == city])
    city_xy[city] = np.array([x_loc, y_loc], dtype=np.float)
    city_data[city] = {'index' : cities.index(city)}
    city_data[city]['x_loc'] = x_loc
    city_data[city]['y_loc'] = y_loc

num_cities = len(cities)

# Create arrays to store demand data
fitness_demand = np.zeros((num_cities,1))
pharmacy_demand = np.zeros((num_cities,1))
physician_demand = np.zeros((num_cities,1))
hotel_demand = np.zeros((num_cities,1))
religion_demand = np.zeros((num_cities,1))
grocery_demand = np.zeros((num_cities,1))
restaurant_demand = np.zeros((num_cities,1))

# Create arrays to store population and related data
devices = np.zeros((num_cities,1))
populations = np.zeros((num_cities,1))
households = np.zeros((num_cities,1))

# Save devices data by city
print('Processing device data...')
device_data = pd.read_excel(os.path.join(data_path, 'DevicesCount_AZ_202004.xlsx'))
for indexDF, rowDF in device_data.iterrows():
    city = geoid.loc[geoid['GEOID'] == rowDF['census_block_group']]['City']
    if len(city.tolist()) != 0:
        ind = cities.index(city.tolist()[0])
        devices[ind] = devices[ind] + rowDF['number_devices_residing']

# Save population data by city
print('Processing population data...')
household_data = pd.read_excel(os.path.join(data_path, 'Househould_Population_blockgroup.xlsx'))
for index, row in household_data.iterrows():
    city = geoid.loc[geoid['GEOID'] == row['GEOID']]['City']
    if len(city.tolist()) != 0:
        ind = cities.index(city.tolist()[0])
        populations[ind] = populations[ind] + row['TOTAL_Population']
        households[ind] = households[ind] + row['TOTAL_Household']

for i in range(num_cities):
    city_data[cities[i]]['devices'] = devices[i]
    city_data[cities[i]]['population'] = populations[i]
    city_data[cities[i]]['households'] = households[i]

# Process grocery data
print('Processing grocery data...')
grocery_data = pd.read_excel(os.path.join(data_path, '202004_IntercityFlow-2.xlsx'))
grocery_demand_dest_mat = np.zeros((num_cities, num_cities))
for indexDF, rowDF in grocery_data.iterrows():
    city_ind = cities.index(rowDF['Origin'])
    city_dest_ind = cities.index(rowDF['Destination'])
    grocery_demand_dest_mat[city_ind, city_dest_ind] = int(grocery_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Sum of Count'] * grocery_freq))

for i in range(num_cities):
    for j in range(num_cities):
        grocery_demand_dest_mat[i,j] = grocery_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    city_data[cities[i]]['grocery_demand_dest'] = grocery_demand_dest_mat[i, :]

for i in range(num_cities):
    grocery_demand[i] = np.sum(grocery_demand_dest_mat[i,:])
    if grocery_demand[i] <= min_demand_val:
        grocery_demand[i] = min_demand_val
    city_data[cities[i]]['grocery_demand'] = grocery_demand[i]

# Process fintess data
print('Processing fitness data...')
fitness_data = pd.read_excel(os.path.join(data_path, 'FitnessCenter_Intercity.xlsx'))
fitness_demand_dest_mat = np.zeros((num_cities, num_cities))
for indexDF, rowDF in fitness_data.iterrows():
    city_ind = cities.index(rowDF['BG_City'])
    city_dest_ind = cities.index(rowDF['POI_City'])
    fitness_demand_dest_mat[city_ind, city_dest_ind] = int(fitness_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * fitness_freq))

for i in range(num_cities):
    for j in range(num_cities):
        fitness_demand_dest_mat[i,j] = fitness_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    city_data[cities[i]]['fitness_demand_dest'] = fitness_demand_dest_mat[i, :]

for i in range(num_cities):
    fitness_demand[i] = np.sum(fitness_demand_dest_mat[i,:])
    if fitness_demand[i] <= min_demand_val:
        fitness_demand[i] = min_demand_val
    city_data[cities[i]]['fitness_demand'] = fitness_demand[i]

# Process pharmacy data
print('Processing pharmacy data...')
pharmacy_data = pd.read_excel(os.path.join(data_path, 'PharmaciesDrugStores_Intercity.xlsx'))
pharmacy_data = pharmacy_data.head(11446) # Cut out garbage rows
pharmacy_demand_dest_mat = np.zeros((num_cities, num_cities))
for indexDF, rowDF in pharmacy_data.iterrows():
    city_ind = cities.index(rowDF['BG_City'])
    city_dest_ind = cities.index(rowDF['POI_City'])
    pharmacy_demand_dest_mat[city_ind, city_dest_ind] = int(pharmacy_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * pharmacy_freq))

for i in range(num_cities):
    for j in range(num_cities):
        pharmacy_demand_dest_mat[i,j] = pharmacy_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    city_data[cities[i]]['pharmacy_demand_dest'] = pharmacy_demand_dest_mat[i, :]

for i in range(num_cities):
    pharmacy_demand[i] = np.sum(pharmacy_demand_dest_mat[i,:])
    if pharmacy_demand[i] <= min_demand_val:
        pharmacy_demand[i] = min_demand_val
    city_data[cities[i]]['pharmacy_demand'] = pharmacy_demand[i]

# Process physician data
print('Processing physician data...')
physician_data = pd.read_excel(os.path.join(data_path, 'Physicians_Intercity.xlsx'))
physician_demand_dest_mat = np.zeros((num_cities, num_cities))
for indexDF, rowDF in physician_data.iterrows():
    city_ind = cities.index(rowDF['BG_City'])
    city_dest_ind = cities.index(rowDF['POI_City'])
    physician_demand_dest_mat[city_ind, city_dest_ind] = int(physician_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * physician_freq))

for i in range(num_cities):
    for j in range(num_cities):
        physician_demand_dest_mat[i,j] = physician_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    city_data[cities[i]]['physician_demand_dest'] = physician_demand_dest_mat[i, :]

for i in range(num_cities):
    physician_demand[i] = np.sum(physician_demand_dest_mat[i,:])
    if physician_demand[i] <= min_demand_val:
        physician_demand[i] = min_demand_val
    city_data[cities[i]]['physician_demand'] = physician_demand[i]

# Process hotel data
print('Processing hotel data...')
hotel_data = pd.read_excel(os.path.join(data_path, 'HotelMotel_Intercity.xlsx'))
hotel_demand_dest_mat = np.zeros((num_cities, num_cities))
for indexDF, rowDF in hotel_data.iterrows():
    city_ind = cities.index(rowDF['BG_City'])
    city_dest_ind = cities.index(rowDF['POI_City'])
    hotel_demand_dest_mat[city_ind, city_dest_ind] = int(hotel_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * hotel_freq))

for i in range(num_cities):
    for j in range(num_cities):
        hotel_demand_dest_mat[i,j] = hotel_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    city_data[cities[i]]['hotel_demand_dest'] = hotel_demand_dest_mat[i, :]

for i in range(num_cities):
    hotel_demand[i] = np.sum(hotel_demand_dest_mat[i,:])
    if hotel_demand[i] <= min_demand_val:
        hotel_demand[i] = min_demand_val
    city_data[cities[i]]['hotel_demand'] = hotel_demand[i]

# # Process religion data
# print('Processing religion data...')
# religion_data = pd.read_excel(os.path.join(data_path, 'ReligiousOrganizations_Intercity.xlsx'))
# religion_demand_dest_mat = np.zeros((num_cities, num_cities))
# for indexDF, rowDF in religion_data.iterrows():
#     city_ind = cities.index(rowDF['BG_City'])
#     city_dest_ind = cities.index(rowDF['POI_City'])
#     religion_demand_dest_mat[city_ind, city_dest_ind] = int(religion_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * religion_freq))

# for i in range(num_cities):
#     for j in range(num_cities):
#         religion_demand_dest_mat[i,j] = religion_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
#     city_data[cities[i]]['religion_demand_dest'] = religion_demand_dest_mat[i, :]

# for i in range(num_cities):
#     religion_demand[i] = np.sum(religion_demand_dest_mat[i,:])
#     if religion_demand[i] <= min_demand_val:
#         religion_demand[i] = min_demand_val
#     city_data[cities[i]]['religion_demand'] = religion_demand[i]

# Process restaurant data
print('Processing restaurant data...')
restaurant_data = pd.read_excel(os.path.join(data_path, 'Restaurant_intercity.xlsx'))
restaurant_demand_dest_mat = np.zeros((num_cities, num_cities))
for indexDF, rowDF in restaurant_data.iterrows():
    if rowDF['POI_City'] in cities:
        city_ind = cities.index(rowDF['BG_City'])
        city_dest_ind = cities.index(rowDF['POI_City'])
        restaurant_demand_dest_mat[city_ind, city_dest_ind] = int(restaurant_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * restaurant_freq))

for i in range(num_cities):
    for j in range(num_cities):
        restaurant_demand_dest_mat[i,j] = restaurant_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    city_data[cities[i]]['restaurant_demand_dest'] = restaurant_demand_dest_mat[i, :]

for i in range(num_cities):
    restaurant_demand[i] = np.sum(restaurant_demand_dest_mat[i,:])
    if restaurant_demand[i] <= min_demand_val:
        restaurant_demand[i] = min_demand_val
    city_data[cities[i]]['restaurant_demand'] = restaurant_demand[i]

# Save the results

# First check if the save directory exists
if not os.path.isdir(os.path.join(data_path, 'data_processing_outputs')):
    os.mkdir(os.path.join(data_path, 'data_processing_outputs'))

demand_array=np.concatenate((grocery_demand, fitness_demand, pharmacy_demand, physician_demand, hotel_demand, restaurant_demand), axis=1)
demand_array.shape

np.save(os.path.join(data_path, 'data_processing_outputs', 'demand_array_arizona.npy'), demand_array)
np.save(os.path.join(data_path, 'data_processing_outputs', 'populations_array_arizona.npy'), populations)

pickle.dump(city_data, open(os.path.join(data_path, 'data_processing_outputs', 'city_data.p'), 'wb'))
