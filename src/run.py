from optimization.covid_gurobi_region_quad_fun_sparse_quad_infect_cost_entity_same import gurobi_solver
from params_config import params
from utils.tester import Tester

tester = Tester()
tester.set_params(params)

solver = gurobi_solver(tester)
solver.run()
