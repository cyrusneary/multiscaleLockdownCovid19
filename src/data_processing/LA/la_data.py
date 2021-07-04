import sys, os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle

base_file_path = os.path.abspath(os.path.join(os.curdir, '..','..', '..')) # should point to the level above the src directory
data_path = os.path.join(base_file_path, 'data', 'IntercityFlow_LA')

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

# Find the average x-y locations of the various cities
print('Processing GEOID data...')

fitness = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Fitness.xlsx'))
grocery = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Grocery.xlsx'))
hotel = pd.read_excel(os.path.join(data_path, 'Intercity_LA_HotelMotel.xlsx'))
pharmacy = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Pharmacy.xlsx'))
physician = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Physician.xlsx'))
religion = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Religious.xlsx'))
restaurant = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Restaurant.xlsx'))

geoid = pd.DataFrame()
geoid['ID'] = pd.concat([fitness['visitor_home_cbgs'],pharmacy['visitor_home_cbgs'],physician['visitor_home_cbgs'],
                    hotel['visitor_home_cbgs'],religion['visitor_home_cbgs'], grocery['visitor_home_cbgs'],
                    restaurant['visitor_home_cbgs']])

geoid['City'] = pd.concat([fitness['City_bg'],pharmacy['City_bg'],physician['City_bg'],
                    hotel['City_bg'],religion['City_bg'], grocery['City_bg'],
                    restaurant['City_bg']])

geoid['X_blockgroup'] = pd.concat([fitness['X_bg'],pharmacy['X_bg'],physician['X_bg'],
                    hotel['X_bg'],religion['X_bg'], grocery['X_bg'],
                    restaurant['X_bg']])

geoid['Y_blockgroup'] = pd.concat([fitness['Y_bg'],pharmacy['Y_bg'],physician['Y_bg'],
                    hotel['Y_bg'],religion['Y_bg'], grocery['Y_bg'],
                    restaurant['Y_bg']])

#geoid = pd.read_excel(os.path.join(data_path, 'GEOID_CityXY.xlsx'))
county_fitness = pd.read_excel(os.path.join(data_path,'CA_Fitness_County.xlsx'))
counties = list(county_fitness.NAME.unique())
counties.remove('Inglewood')
counties.remove('Silverado')
counties.remove('Compton')
counties.remove('North Antelope Valley')
counties.remove('South Antelope Valley')
counties.remove('Downey-Norwalk')
num_counties = len(counties)
cities = list(geoid.City.unique())
#cities.append('Hunts Point')
county_data = dict()
city_xy = dict()
for county in counties:
    #x_loc = np.average(geoid.X_blockgroup[geoid.City == city])
    #y_loc = np.average(geoid.Y_blockgroup[geoid.City == city])
    #city_xy[city] = np.array([x_loc, y_loc], dtype=np.float)
    county_data[county] = {'index' : counties.index(county)}
    #county_data[city]['x_loc'] = x_loc
    #county_data[city]['y_loc'] = y_loc
print(counties)
num_cities = len(cities)



loc_counties = []
for loc in cities:
    id = fitness.loc[fitness['City_bg'] == loc]['safegraph_place_id']
    county = county_fitness.loc[id.tolist()[0] == county_fitness['safegraph_']]['NAME']
    loc_counties.append(county.tolist()[0])
    
county_dict = dict(zip(cities,loc_counties))

# Create arrays to store demand data
fitness_demand = np.zeros((num_counties,1))
pharmacy_demand = np.zeros((num_counties,1))
physician_demand = np.zeros((num_counties,1))
hotel_demand = np.zeros((num_counties,1))
religion_demand = np.zeros((num_counties,1))
grocery_demand = np.zeros((num_counties,1))
restaurant_demand = np.zeros((num_counties,1))

# Create arrays to store population and related data
devices = np.zeros((num_counties,1))
populations = np.zeros((num_counties,1))
households = np.zeros((num_counties,1))

# Save devices data by city
print('Processing device data...')
device_data = pd.read_excel(os.path.join(data_path, 'LA_Devices_bg.xlsx'))
for indexDF, rowDF in device_data.iterrows():
    city = geoid.loc[geoid['ID'] == rowDF['census_block_group']]['City']
    if len(city.tolist()) != 0:
        county = county_dict.get(city.tolist()[0])
        ind = counties.index(county)
        devices[ind] = devices[ind] + rowDF['number_devices_residing']

# Save population data by city
print('Processing population data...')
household_data = pd.read_excel(os.path.join(data_path, 'Population_bg_LA.xlsx'))
for index, row in household_data.iterrows():
    city = geoid.loc[geoid['ID'] == row['GEO_ID']]['City']
    if len(city.tolist()) != 0:
        county = county_dict.get(city.tolist()[0])
        ind = counties.index(county)
        populations[ind] = populations[ind] + row['B01001_001E']
#        households[ind] = households[ind] + row['TOTAL_Household']

#for i in range(num_counties):
#    city_data[counties[i]]['devices'] = devices[i]
#    city_data[counties[i]]['population'] = populations[i]
#    city_data[cities[i]]['households'] = households[i]

# Process grocery data
print('Processing grocery data...')
grocery_data = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Grocery.xlsx'))
grocery_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in grocery_data.iterrows():
    if rowDF['City_poi'] in cities:
        bg_county = county_dict.get(rowDF['City_bg'])
        poi_county = county_dict.get(rowDF['City_poi'])
        bg_ind = counties.index(bg_county)
        poi_ind = counties.index(poi_county)
        grocery_demand_dest_mat[bg_ind, poi_ind] = int(grocery_demand_dest_mat[bg_ind, poi_ind] + (rowDF['Count'] * grocery_freq))

for i in range(num_counties):
    for j in range(num_counties):
        grocery_demand_dest_mat[i,j] = grocery_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['grocery_demand_dest'] = grocery_demand_dest_mat[i, :]

for i in range(num_counties):
    grocery_demand[i] = np.sum(grocery_demand_dest_mat[i,:])
    if grocery_demand[i] <= min_demand_val:
        grocery_demand[i] = min_demand_val
    county_data[counties[i]]['grocery_demand'] = grocery_demand[i]


# Process fintess data
print('Processing fitness data...')
fitness_data = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Fitness.xlsx'))
fitness_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in fitness_data.iterrows():
    bg_county = county_dict.get(rowDF['City_bg'])
    poi_county = county_dict.get(rowDF['City_poi'])
    if bg_county in counties and poi_county in counties:
        bg_ind = counties.index(bg_county)
        poi_ind = counties.index(poi_county)
    fitness_demand_dest_mat[bg_ind, poi_ind] = int(fitness_demand_dest_mat[bg_ind, poi_ind] + (rowDF['Count'] * fitness_freq))

for i in range(num_counties):
    for j in range(num_counties):
        fitness_demand_dest_mat[i,j] = fitness_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['fitness_demand_dest'] = fitness_demand_dest_mat[i, :]

for i in range(num_counties):
    fitness_demand[i] = np.sum(fitness_demand_dest_mat[i,:])
    if fitness_demand[i] <= min_demand_val:
        fitness_demand[i] = min_demand_val
    county_data[counties[i]]['fitness_demand'] = fitness_demand[i]

# Process pharmacy data
print('Processing pharmacy data...')
pharmacy_data = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Pharmacy.xlsx'))
pharmacy_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in pharmacy_data.iterrows():
    bg_county = county_dict.get(rowDF['City_bg'])
    poi_county = county_dict.get(rowDF['City_poi'])
    if bg_county in counties and poi_county in counties:
        bg_ind = counties.index(bg_county)
        poi_ind = counties.index(poi_county)
    pharmacy_demand_dest_mat[bg_ind, poi_ind] = int(pharmacy_demand_dest_mat[bg_ind, poi_ind] + (rowDF['Count'] * pharmacy_freq))

for i in range(num_counties):
    for j in range(num_counties):
        pharmacy_demand_dest_mat[i,j] = pharmacy_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['pharmacy_demand_dest'] = pharmacy_demand_dest_mat[i, :]

for i in range(num_counties):
    pharmacy_demand[i] = np.sum(pharmacy_demand_dest_mat[i,:])
    if pharmacy_demand[i] <= min_demand_val:
        pharmacy_demand[i] = min_demand_val
    county_data[counties[i]]['pharmacy_demand'] = pharmacy_demand[i]

# Process physician data
print('Processing physician data...')
physician_data = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Physician.xlsx'))
physician_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in physician_data.iterrows():
    bg_county = county_dict.get(rowDF['City_bg'])
    poi_county = county_dict.get(rowDF['City_poi'])
    if bg_county in counties and poi_county in counties:
        bg_ind = counties.index(bg_county)
        poi_ind = counties.index(poi_county)
    physician_demand_dest_mat[bg_ind, poi_ind] = int(physician_demand_dest_mat[bg_ind, poi_ind] + (rowDF['Count'] * physician_freq))

for i in range(num_counties):
    for j in range(num_counties):
        physician_demand_dest_mat[i,j] = physician_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['physician_demand_dest'] = physician_demand_dest_mat[i, :]

for i in range(num_counties):
    physician_demand[i] = np.sum(physician_demand_dest_mat[i,:])
    if physician_demand[i] <= min_demand_val:
        physician_demand[i] = min_demand_val
    county_data[counties[i]]['physician_demand'] = physician_demand[i]

# Process hotel data
print('Processing hotel data...')
hotel_data = pd.read_excel(os.path.join(data_path, 'Intercity_LA_HotelMotel.xlsx'))
hotel_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in hotel_data.iterrows():
    bg_county = county_dict.get(rowDF['City_bg'])
    poi_county = county_dict.get(rowDF['City_poi'])
    if bg_county in counties and poi_county in counties:
        bg_ind = counties.index(bg_county)
        poi_ind = counties.index(poi_county)
    hotel_demand_dest_mat[bg_ind, poi_ind] = int(hotel_demand_dest_mat[bg_ind, poi_ind] + (rowDF['Count'] * hotel_freq))

for i in range(num_counties):
    for j in range(num_counties):
        hotel_demand_dest_mat[i,j] = hotel_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['hotel_demand_dest'] = hotel_demand_dest_mat[i, :]

for i in range(num_counties):
    hotel_demand[i] = np.sum(hotel_demand_dest_mat[i,:])
    if hotel_demand[i] <= min_demand_val:
        hotel_demand[i] = min_demand_val
    county_data[counties[i]]['hotel_demand'] = hotel_demand[i]

# Process religion data
# print('Processing religion data...')
# religion_data = pd.read_excel(os.path.join(data_path, 'Intercity_Seattle_Religious.xlsx'))
# religion_demand_dest_mat = np.zeros((num_cities, num_cities))
# for indexDF, rowDF in religion_data.iterrows():
#     city_ind = cities.index(rowDF['City_bg'])
#     city_dest_ind = cities.index(rowDF['City_poi'])
#     religion_demand_dest_mat[city_ind, city_dest_ind] = int(religion_demand_dest_mat[city_ind, city_dest_ind] + (rowDF['Count'] * religion_freq))
#
# for i in range(num_cities):
#     for j in range(num_cities):
#         religion_demand_dest_mat[i,j] = religion_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
#     city_data[cities[i]]['religion_demand_dest'] = religion_demand_dest_mat[i, :]
#
# for i in range(num_cities):
#     religion_demand[i] = np.sum(religion_demand_dest_mat[i,:])
#     if religion_demand[i] <= min_demand_val:
#         religion_demand[i] = min_demand_val
#     city_data[cities[i]]['religion_demand'] = religion_demand[i]

# Process restaurant data
print('Processing restaurant data...')
restaurant_data = pd.read_excel(os.path.join(data_path, 'Intercity_LA_Restaurant.xlsx'))
restaurant_demand_dest_mat = np.zeros((num_counties, num_counties))
for indexDF, rowDF in restaurant_data.iterrows():
    if rowDF['City_poi'] in cities:
        bg_county = county_dict.get(rowDF['City_bg'])
        poi_county = county_dict.get(rowDF['City_poi'])
        if bg_county in counties and poi_county in counties:
            bg_ind = counties.index(bg_county)
            poi_ind = counties.index(poi_county)
        restaurant_demand_dest_mat[bg_ind, poi_ind] = int(restaurant_demand_dest_mat[bg_ind, poi_ind] + (rowDF['Count'] * restaurant_freq))

for i in range(num_counties):
    for j in range(num_counties):
        restaurant_demand_dest_mat[i,j] = restaurant_demand_dest_mat[i,j] * populations[i] / devices[i] * month_day_time_conversion
    county_data[counties[i]]['restaurant_demand_dest'] = restaurant_demand_dest_mat[i, :]

for i in range(num_counties):
    restaurant_demand[i] = np.sum(restaurant_demand_dest_mat[i,:])
    if restaurant_demand[i] <= min_demand_val:
        restaurant_demand[i] = min_demand_val
    county_data[counties[i]]['restaurant_demand'] = restaurant_demand[i]

# Save the results

# First check if the save directory exists
if not os.path.isdir(os.path.join(data_path, 'data_processing_outputs')):
    os.mkdir(os.path.join(data_path, 'data_processing_outputs'))

demand_array=np.concatenate((grocery_demand, fitness_demand, pharmacy_demand, physician_demand, hotel_demand, restaurant_demand), axis=1)
demand_array.shape

np.save(os.path.join(data_path, 'data_processing_outputs', 'demand_array_la.npy'), demand_array)
np.save(os.path.join(data_path, 'data_processing_outputs', 'populations_array_la.npy'), populations)

pickle.dump(county_data, open(os.path.join(data_path, 'data_processing_outputs', 'city_data.p'), 'wb'))