from optimization.heterogeneous_optimization import gurobi_solver
from params_config import params
from utils.tester import Tester

tester = Tester()
tester.set_params(params)

solver = gurobi_solver(tester)
solver.run()
