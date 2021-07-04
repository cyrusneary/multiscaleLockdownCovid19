# multiscaleLockdownCovid19

This repository contains the data and code necessary to reproduce the results of the project "Multiscale Heterogeneous Optimal Lockdown Control for COVID-19 Using Geographic Information".

## Dependencies
To run the code in this repository, Python>=3.8 is required as well as the following packages:
- gurobipy
- networkx
- pickle
- matplotlib
- tikzplotlib
- pandas

## Contents

### /data
This folder contains all of the data used to generate the multiscale SIRD model for all six of the considered MSAs.

#### /data/output_data_tracking.ods
This excel file contains a description of the results for all of the cases in the project with detailed explanations.

### /src
This folder contains all of the code used to generate the lockdown policies presented in the project for all six of the considered MSAs. 

#### /src/run.py
This file is used to run the experiments of the project. It loads the configuration for the experiment from src/params_config.py, and calls src/optimization/heterogeneous_optimization.py.

#### /src/params_config.py
This file is used to configure the experiment being run.

#### /src/data_processing
This folder contains the code used to process the data in /data to generate numpy arrays that can be loaded directly into the relevant optimization code. To generate all such necessary arrays, run 1. src/data_processing/MSA_PATH/MSA_data.py, 2. src/data_processing/MSA_PATH/generate_MSA_adjacency_matrix.py, and 3. src/data_processing/MSA_PATH/generate_MSA_edge_weights.py. Here, MSA and MSA_PATH should be replaced with the appropriate name for the MSA in question. The outputs of these python files are stored in /data/MSA_PATH/data_processing_outputs.
 
#### /src/optimization
The optimization problem is implemented in /src/optimization/heterogeneous_optimization.py. The pickled output dictionaries containing the results of the experiments are stored in /src/optimization/save.

#### /src/plotting
This folder contains all of the code used to generate the plots in the project manuscript. 
