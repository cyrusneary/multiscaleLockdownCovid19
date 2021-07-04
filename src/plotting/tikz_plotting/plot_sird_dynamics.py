
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
# save_file_name = '2021-03-18_12-12-22' # No lockdown
# save_file_name = '2021-03-20_03-48-37' # nominal lockdown
# save_file_name = '2021-04-09_21-26-07' # nominal lockdown (updated)
# save_file_name = '2021-04-02_04-20-41' # VERY low lockdown cost
save_file_name = '2021-04-23_14-02-29'

# city_name = 'Seattle'
# save_file_name = '2021-04-26_18-03-56' # nominal lockdown
# save_file_name = '2021-03-21_13-01-03' # no lockdown

# city_name = 'Dallas'
# save_file_name =  '2021-03-29_16-45-38' # No lockdown
# save_file_name = '2021-04-24_18-50-03' # nominal settings

# city_name = 'LA'
# save_file_name =  '2021-04-23_22-43-09' # nominal settings
# save_file_name = '2021-04-09_09-39-36' # No lockdown

# city_name = 'Chicago'
# save_file_name =  '2021-04-26_09-12-01' # nominal settings
# save_file_name = '2021-04-08_14-53-52' # No lockdown

# city_name = 'NY'
# # save_file_name =  '2021-04-25_15-53-19' # nominal settings
# save_file_name = '2021-04-09_09-40-36' # No lockdown

fontsize = 30
legend_fontsize = 40

base_directory = os.getcwd()
base_directory = base_directory[0:base_directory.find('src')+3]

# %%

file_path = os.path.join(base_directory, 'optimization', 'save', save_file_name)

with open(file_path,'rb') as f:
    tester = pickle.load(f)

# %%

num_city = tester.params['m']
num_time = tester.params['n']
num_entity = tester.params['num_entity']
phi_val = tester.results['phi_best']
scale_frac = tester.params['scale_frac']
cost_param=tester.params['econ_param']
print(cost_param,save_file_name)
t = np.arange(0, num_time)

I = tester.results['I_best'] * scale_frac
S = tester.results['S_best'] * scale_frac
D = tester.results['D_best'] * scale_frac
R = tester.results['R_best'] * scale_frac

# %%

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, np.sum(S, axis=1) / 1e6, color='blue', linewidth=4, label='Susceptible')
ax.plot(t, np.sum(I, axis=1) / 1e6, color='red', linewidth=4, label='Infected')
ax.plot(t, np.sum(R, axis=1) / 1e6, color='green', linewidth=4, label='Recovered')
ax.plot(t, np.sum(D, axis=1) / 1e6, color='magenta', linewidth=4, label='Dead')
ax.grid()
ax.ticklabel_format(axis='y', style='sci')
ax.tick_params(axis='both', labelsize=fontsize)
print(np.sum(R, axis=1))
ax.set_xlim(0, num_time - 1)

ax.set_xlabel('Time [days]')
ax.set_ylabel('Millions of people')

ax.legend()

tikz_save_file_path = os.path.join(base_directory, 'plotting', 'tikz_plotting', city_name)
filename = os.path.join(tikz_save_file_path, 'scale_cost_by_pop_nominal_lockdown_sird_dynamics.tex')
tikzplotlib.save(filename)
# %%
