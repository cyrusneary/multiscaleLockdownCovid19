
import os
import numpy
params = dict()

# Handle relative file paths - hardcode this to '<project_path>/src' if the below code isn't working for you.
base_directory = os.getcwd()
if base_directory.find('src') == -1:
    # assume the person is in the directory one level above src
    base_directory = os.path.join(base_directory, 'src')
else:
    # otherwise, if the user is calling the code from deeper in the project structure, bring them back to src
    base_directory = base_directory[0:base_directory.find('src') + 3]


base_directory_covid=(os.path.dirname(os.getcwd()))

# file paths
params['entity_mat_path'] = os.path.join(base_directory_covid, 'data','IntercityFlow_Seattle', 'data_processing_outputs', 'demand_array_seattle.npy')
params['population_mat_path'] = os.path.join(base_directory_covid,'data','IntercityFlow_Seattle', 'data_processing_outputs', 'populations_array_seattle.npy')
params['inection_mat_path'] = os.path.join(base_directory_covid,  'data', 'IntercityFlow_Seattle', 'data_processing_outputs','seattle_cases.npy')
params['adj_mat_path'] = os.path.join(base_directory_covid,'data','IntercityFlow_Seattle', 'data_processing_outputs', 'adjacency_matrix.npy')
params['edge_weights'] = os.path.join(base_directory_covid,'data','IntercityFlow_Seattle', 'data_processing_outputs', 'edge_weights.npy')
pop_mat = numpy.load(params['population_mat_path'])
#print(numpy.size(pop_mat))
params['geo_adj_mat_path'] = os.path.join(base_directory_covid,'data','Chicago_Intercounty_Flow', 'data_processing_outputs', 'adjacency_matrix.npy')

params['save_file_path'] = os.path.join(base_directory, 'optimization', 'save')

# model parameters
params['m'] = numpy.size(pop_mat)
params['adj_region'] = 4
params['num_entity'] = 6
params['econ_param'] = 2e-4 #parameter for cost of lockdown
params['timer_val'] = 10 #setting the time interval for changing the policies

params['beta'] = 0.45 #initial parameter values for infection
params['gamma'] = 0.06
params['Gamma_linear'] = 0.002

params['Ts'] = 1 #sampling time
params['n'] = 100 # time horizon
params['num_iter'] = 100 # 100 # number of interations

params['scale_frac'] = 1e3 #scaling down population by 1000 for better numerical purposes, does not change the method

params['online_scale'] = 1 #scaling factor for online shoppers
params['capacity_scale'] = 2 #capacity is 1.5x the demand

params['mu_cons'] = 1e4 #penalty for violating the constraints

params['trust_region'] = 150 # initializing trust region
params['min_phi_val'] = 0.05 # min phi for self loop

params['low_pov_constant']=0.4 #minimum allowed capacity for low poverty cities

params['high_pov_constant']=0.6 #minimum allowed capacity for low poverty cities

params['mobility_restriction']=5 # for the "mobility restriction (tranportation)" version of the problem
