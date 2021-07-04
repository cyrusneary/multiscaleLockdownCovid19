# %%
import sys, os
import pandas as pd
import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
import pickle

base_file_path = os.path.abspath(os.path.join(os.curdir, '..','..', '..')) # should point to the level above the src directory
data_path = os.path.join(base_file_path, 'data', 'Intercity_Dallas')

# (grocery_demand, fitness_demand, pharmacy_demand, physician_demand, hotel_demand, religion_demand, restaurant_demand)
# Entity indexes
# 0 - groceries
# 1 - fitness
# 2 - pharmacy
# 3 - physician
# 4 - hotel
# 5 - religion
# 6 - restaurant

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

# %%

# First get a list of the counties in Dallas MSA
county_fitness = pd.read_excel(os.path.join(data_path,'TX_Fitness_County.xlsx'))
counties = list(county_fitness.CNTY_NM.unique())
num_counties = len(counties)
print(counties)
county_data = dict()
for county in counties:
    county_data[county] = {'index' : counties.index(county)}

# %%
# In county data, save a list of the block groups belonging to each county.
for county in counties:
    county_data[county]['bg_list'] = set()

# Load and store block-group statistics
bg_info = dict()

# Save population data by county
print('Processing population data...')
population_data = pd.read_excel(os.path.join(data_path, 'Population_bg_Dallas.xlsx'))
for index, row in population_data.iterrows():
    county = row['NAME']
    if county in counties:
        bg_id = row['GEO_ID']
        population = row['Population']

        bg_info[bg_id] = dict()
        bg_info[bg_id]['county'] = county
        bg_info[bg_id]['population'] = population

        county_data[county]['bg_list'].add(bg_id)

# Save devices data by county
print('Processing device data...')
device_data = pd.read_excel(os.path.join(data_path, 'TX_Devices_bg.xlsx'))
for index, row in device_data.iterrows():
    bg_id = row['census_block_group']
    if bg_id in bg_info.keys():
        devices = row['number_devices_residing']
        bg_info[bg_id]['devices'] = devices

# %%

# Create arrays to store population and related data
devices = np.zeros((num_counties,))
populations = np.zeros((num_counties,))

# Now save populations and device counts by county
for county in counties:
    county_data[county]['population'] = 0
    county_data[county]['devices'] = 0

    # Iterate over the block groups in each county and add the population and device count
    for bg_id in county_data[county]['bg_list']:
        county_data[county]['population'] = county_data[county]['population'] + bg_info[bg_id]['population']
        county_data[county]['devices'] = county_data[county]['devices'] + bg_info[bg_id]['devices']

    devices[county_data[county]['index']] = county_data[county]['devices']
    populations[county_data[county]['index']] = county_data[county]['population']

# %%

# Create a map from safegraph ID to county
sgid_to_county = dict()

fitness_county = pd.read_excel(os.path.join(data_path, 'TX_Fitness_County.xlsx'))
for index, row in fitness_county.iterrows():
    sgid = row['safegraph_']
    county = row['CNTY_NM']
    sgid_to_county[sgid] = county

grocery_county = pd.read_excel(os.path.join(data_path, 'TX_Grocery_County.xlsx'))
for index, row in grocery_county.iterrows():
    sgid = row['safegraph_']
    county = row['CNTY_NM']
    sgid_to_county[sgid] = county

hmotel_county = pd.read_excel(os.path.join(data_path, 'TX_HMotel_County.xlsx'))
for index, row in hmotel_county.iterrows():
    sgid = row['safegraph_']
    county = row['CNTY_NM']
    sgid_to_county[sgid] = county

pharmacy_county = pd.read_excel(os.path.join(data_path, 'TX_Pharmacy_County.xlsx'))
for index, row in pharmacy_county.iterrows():
    sgid = row['safegraph_']
    county = row['CNTY_NM']
    sgid_to_county[sgid] = county

physician_county = pd.read_excel(os.path.join(data_path, 'TX_Physician_County.xlsx'))
for index, row in physician_county.iterrows():
    sgid = row['safegraph_']
    county = row['CNTY_NM_1']
    sgid_to_county[sgid] = county

restaurant_county = pd.read_excel(os.path.join(data_path, 'TX_Restaurant_County.xlsx'))
for index, row in restaurant_county.iterrows():
    sgid = row['safegraph_']
    county = row['CNTY_NM']
    sgid_to_county[sgid] = county

# %%
# Create arrays to store demand data
fitness_demand = np.zeros((num_counties,1))
pharmacy_demand = np.zeros((num_counties,1))
physician_demand = np.zeros((num_counties,1))
hotel_demand = np.zeros((num_counties,1))
religion_demand = np.zeros((num_counties,1))
grocery_demand = np.zeros((num_counties,1))
restaurant_demand = np.zeros((num_counties,1))

# %%

# Process grocery data
print('Processing grocery data...')
grocery_data = pd.read_excel(os.path.join(data_path, 'Intercity_Dallas_Grocery.xlsx'))
grocery_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in grocery_data.iterrows():
    sgid = rowDF['safegraph_place_id']
    destination_county = sgid_to_county[sgid]
    origin_county = bg_info[rowDF['visitor_home_cbgs']]['county']
    count = rowDF['Count']

    destination_ind = county_data[destination_county]['index']
    origin_ind = county_data[origin_county]['index']

    grocery_demand_dest_mat[origin_ind, destination_ind] = \
        int(grocery_demand_dest_mat[origin_ind, destination_ind] + (count * grocery_freq))

for i in range(num_counties):
    for j in range(num_counties):
        grocery_demand_dest_mat[i,j] = grocery_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['grocery_demand_dest'] = grocery_demand_dest_mat[i, :]

for i in range(num_counties):
    grocery_demand[i] = np.sum(grocery_demand_dest_mat[i,:])
    if grocery_demand[i] <= min_demand_val:
        grocery_demand[i] = min_demand_val
    county_data[counties[i]]['grocery_demand'] = grocery_demand[i]

# %%
# Process fintess data
print('Processing fitness data...')
fitness_data = pd.read_excel(os.path.join(data_path, 'Intercity_Dallas_Fitness.xlsx'))
fitness_demand_dest_mat = np.zeros((num_counties, num_counties))

for indexDF, rowDF in fitness_data.iterrows():
    sgid = rowDF['safegraph_place_id']
    destination_county = sgid_to_county[sgid]
    origin_county = bg_info[rowDF['visitor_home_cbgs']]['county']
    count = rowDF['Count']

    destination_ind = county_data[destination_county]['index']
    origin_ind = county_data[origin_county]['index']

    fitness_demand_dest_mat[origin_ind, destination_ind] = \
        int(fitness_demand_dest_mat[origin_ind, destination_ind] + (count * fitness_freq))

for i in range(num_counties):
    for j in range(num_counties):
        fitness_demand_dest_mat[i,j] = fitness_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['fitness_demand_dest'] = fitness_demand_dest_mat[i, :]

for i in range(num_counties):
    fitness_demand[i] = np.sum(fitness_demand_dest_mat[i,:])
    if fitness_demand[i] <= min_demand_val:
        fitness_demand[i] = min_demand_val
    county_data[counties[i]]['fitness_demand'] = fitness_demand[i]

# %%
# Process pharmacy data
print('Processing pharmacy data...')
pharmacy_data = pd.read_excel(os.path.join(data_path, 'Intercity_Dallas_Pharmacy.xlsx'))
pharmacy_demand_dest_mat = np.zeros((num_counties, num_counties))

for indexDF, rowDF in pharmacy_data.iterrows():
    sgid = rowDF['safegraph_place_id']
    destination_county = sgid_to_county[sgid]
    origin_county = bg_info[rowDF['visitor_home_cbgs']]['county']
    count = rowDF['Count']

    destination_ind = county_data[destination_county]['index']
    origin_ind = county_data[origin_county]['index']

    pharmacy_demand_dest_mat[origin_ind, destination_ind] = \
        int(pharmacy_demand_dest_mat[origin_ind, destination_ind] + (count * pharmacy_freq))

for i in range(num_counties):
    for j in range(num_counties):
        pharmacy_demand_dest_mat[i,j] = pharmacy_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['pharmacy_demand_dest'] = pharmacy_demand_dest_mat[i, :]

for i in range(num_counties):
    pharmacy_demand[i] = np.sum(pharmacy_demand_dest_mat[i,:])
    if pharmacy_demand[i] <= min_demand_val:
        pharmacy_demand[i] = min_demand_val
    county_data[counties[i]]['pharmacy_demand'] = pharmacy_demand[i]

# %%

# Process physician data
print('Processing physician data...')
physician_data = pd.read_excel(os.path.join(data_path, 'Intercity_Dallas_Physician.xlsx'))
physician_demand_dest_mat = np.zeros((num_counties, num_counties))

for indexDF, rowDF in physician_data.iterrows():
    sgid = rowDF['safegraph_place_id']
    destination_county = sgid_to_county[sgid]
    origin_county = bg_info[rowDF['visitor_home_cbgs']]['county']
    count = rowDF['Count']

    destination_ind = county_data[destination_county]['index']
    origin_ind = county_data[origin_county]['index']

    physician_demand_dest_mat[origin_ind, destination_ind] = \
        int(physician_demand_dest_mat[origin_ind, destination_ind] + (count * physician_freq))

for i in range(num_counties):
    for j in range(num_counties):
        physician_demand_dest_mat[i,j] = physician_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['physician_demand_dest'] = physician_demand_dest_mat[i, :]

for i in range(num_counties):
    physician_demand[i] = np.sum(physician_demand_dest_mat[i,:])
    if physician_demand[i] <= min_demand_val:
        physician_demand[i] = min_demand_val
    county_data[counties[i]]['physician_demand'] = physician_demand[i]

# %%

# Process hotel data
print('Processing hotel data...')
hotel_data = pd.read_excel(os.path.join(data_path, 'Intercity_Dallas_HotelMotel.xlsx'))
hotel_demand_dest_mat = np.zeros((num_counties, num_counties))

for indexDF, rowDF in hotel_data.iterrows():
    sgid = rowDF['safegraph_place_id']
    destination_county = sgid_to_county[sgid]
    origin_county = bg_info[rowDF['visitor_home_cbgs']]['county']
    count = rowDF['Count']

    destination_ind = county_data[destination_county]['index']
    origin_ind = county_data[origin_county]['index']

    hotel_demand_dest_mat[origin_ind, destination_ind] = \
        int(hotel_demand_dest_mat[origin_ind, destination_ind] + (count * hotel_freq))

for i in range(num_counties):
    for j in range(num_counties):
        hotel_demand_dest_mat[i,j] = hotel_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['hotel_demand_dest'] = hotel_demand_dest_mat[i, :]

for i in range(num_counties):
    hotel_demand[i] = np.sum(hotel_demand_dest_mat[i,:])
    if hotel_demand[i] <= min_demand_val:
        hotel_demand[i] = min_demand_val
    county_data[counties[i]]['hotel_demand'] = hotel_demand[i]

# %%

# Process restaurant data
print('Processing restaurant data...')
restaurant_data = pd.read_excel(os.path.join(data_path, 'Intercity_Dallas_Restaurant.xlsx'))
restaurant_demand_dest_mat = np.zeros((num_counties, num_counties))

for indexDF, rowDF in restaurant_data.iterrows():
    sgid = rowDF['safegraph_place_id']
    destination_county = sgid_to_county[sgid]
    origin_county = bg_info[rowDF['visitor_home_cbgs']]['county']
    count = rowDF['Count']

    destination_ind = county_data[destination_county]['index']
    origin_ind = county_data[origin_county]['index']

    restaurant_demand_dest_mat[origin_ind, destination_ind] = \
        int(restaurant_demand_dest_mat[origin_ind, destination_ind] + (count * restaurant_freq))

for i in range(num_counties):
    for j in range(num_counties):
        restaurant_demand_dest_mat[i,j] = restaurant_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['restaurant_demand_dest'] = restaurant_demand_dest_mat[i, :]

for i in range(num_counties):
    restaurant_demand[i] = np.sum(restaurant_demand_dest_mat[i,:])
    if restaurant_demand[i] <= min_demand_val:
        restaurant_demand[i] = min_demand_val
    county_data[counties[i]]['restaurant_demand'] = restaurant_demand[i]

# %%
# Save the results

# First check if the save directory exists
if not os.path.isdir(os.path.join(data_path, 'data_processing_outputs')):
    os.mkdir(os.path.join(data_path, 'data_processing_outputs'))

demand_array=np.concatenate((grocery_demand, fitness_demand, pharmacy_demand, physician_demand, hotel_demand, restaurant_demand), axis=1)
demand_array.shape
print(demand_array)
np.save(os.path.join(data_path, 'data_processing_outputs', 'demand_array_dallas.npy'), demand_array)
np.save(os.path.join(data_path, 'data_processing_outputs', 'populations_array_dallas.npy'), populations)

pickle.dump(county_data, open(os.path.join(data_path, 'data_processing_outputs', 'county_data.p'), 'wb'))

# %%
