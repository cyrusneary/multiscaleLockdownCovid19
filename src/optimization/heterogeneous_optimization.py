from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
import scipy.io
import math
import pickle
import datetime
import os

#main class
class gurobi_solver():
    #function to initialize the data
    def __init__(self, tester):

        # get the problem parameters to initialize the problem
        params = tester.params
        self.tester = tester

        #number of regions
        self.m = params['m']

        #loading up the date
        self.entity_mat = np.load(params['entity_mat_path'])
        self.pop_mat = np.load(params['population_mat_path'])
        self.infection_mat = np.load(params['inection_mat_path'])

        self.adj_mat = np.load(params['adj_mat_path'])
        #number of adjacent regions per city
        self.adj_region = 3
        #number of entities
        self.num_entity = params['num_entity']
        print(self.num_entity,"num_entity")
        print(self.adj_mat)
        #creating the adjacent cities for each city
        self.adj_list=[]
        for i in range(self.m):
            for j in range(self.m):
                if self.adj_mat[i][j]>0.5:
                    self.adj_list.append((i,j))

        #parameter for cost of lockdown
        self.econ_param = params['econ_param']

        #setting the time interval for changing the policies
        self.timer_val = params['timer_val']

        #initial parameter values for infection
        self.beta = params['beta']
        self.gamma = params['gamma']
        self.Gamma_linear = params['Gamma_linear']

        self.weight_mat=np.load(params['edge_weights'])
        #print(self.weight_mat)
        #weight for the graph, initialized with uniform
        self.weight=np.zeros((self.m,self.m,self.num_entity))
        for i in range(self.m):
            for j in range(self.m):
                for k in range(self.num_entity):
                    self.weight[i][j][k]=self.weight_mat[i][j][k]

        self.high_pov_cons=params['high_pov_constant']
        self.low_pov_cons=params['low_pov_constant']

        #sampling time
        self.Ts = params['Ts']

        #number of time horizon
        self.n = params['n']

        #number of iteration
        self.num_iter = params['num_iter']

        self.min_phi_val = params['min_phi_val']

        #initial matrices for the variables
        self.I = np.zeros((self.n,self.m))
        self.S = np.zeros((self.n,self.m))
        self.R = np.zeros((self.n,self.m))
        self.D = np.zeros((self.n,self.m))
        self.N = np.zeros((self.n,self.m))
        self.A = np.zeros((self.n,self.m,self.num_entity))
        self.B = np.zeros((self.n,self.m,self.num_entity))

        #initial value of L function
        self.L= np.ones((self.n,self.m,self.num_entity))
        #initial matrix for the phi variables
        self.phi = np.ones((self.n,self.m,self.m))/self.m

        #rather random way to initialize the phi, 0.668 is
        # to take into account the online shopping where everything is open
        for t in range(self.n - 1):
            for i in range(self.m):
                self.phi[t][i][i]=0.9
                for j in range(self.m):
                    if not j==i:
                        self.phi[t][i][j] = (1-0.9)/(self.adj_region-1)

        #initializing the values of infections, susceptibles, recovereds and deaths
        self.Iinit=np.reshape(self.infection_mat[0:self.m],(self.m,))
        for i in range(len(self.Iinit)):
            if self.Iinit[i]==0:
                self.Iinit[i]=5
        self.Hinit=np.reshape(self.infection_mat[0:self.m],(self.m,))
        self.Sinit=np.reshape(self.pop_mat[0:self.m],(self.m,))
        self.Rinit=np.zeros((self.m,))
        self.Dinit=np.zeros((self.m,))

        #scaling down population by 1000 for better numerical purposes, does not change the method
        self.scale_fac=1e3

        #setting initial values
        self.I[0][:] = self.Iinit/self.scale_fac
        self.R[0][:] = self.Rinit/self.scale_fac
        self.D[0][:] = self.Dinit/self.scale_fac
        self.S[0][:] = self.Sinit/self.scale_fac

        #total of people
        self.Ntot=(self.Iinit+self.Rinit+self.Sinit)/self.scale_fac

        #setting up the rho, larger for larger citires
        self.rho = np.zeros((self.m,))
        maxval=max(self.Ntot)
        for i in range(self.m):
            self.rho[i]=(0.36+0.00*self.Ntot[i]/maxval)/self.beta
        #print("printing rho",self.rho)
        print("printing total population",self.Ntot)
        print("printing I,R,S",self.Iinit,self.Rinit,self.Sinit)

        #setting up capacity and demand using data
        self.Capacity = np.zeros((self.m,self.num_entity))
        self.demand = np.zeros((self.m,self.num_entity))
        self.demand_tot = np.zeros((self.m))
        self.demand_mat=np.load(params['entity_mat_path'])

        for l in range(self.num_entity):
            self.demand_tot[:]+=np.reshape(self.demand_mat[0:self.m,l],(self.m,))
            print(self.demand_tot)
        for i in range(self.m):	        
            for l in range(self.num_entity):
                self.demand[i,l]=(self.demand_mat[i,l])*self.pop_mat[i]/self.demand_tot[i]/self.scale_fac
        print(self.demand[:,:])
        #capacity is 1.5x the demand
        self.capacity_scale = params['capacity_scale']
        for l in range(self.num_entity):
            self.Capacity[:,l] = self.demand[:,l]*self.capacity_scale

        #setting up A and B matrices, online_scale is here is to limit the effectiveness of lockdown
        self.online_scale = params['online_scale']
        for t in range(self.n):
            for l in range(self.num_entity):
                self.A[t,:,l] = -0.668*self.demand[:,l]/self.online_scale
                self.B[t,:,l] = self.demand[:,l]/self.online_scale

        #penalty for violating the constraints
        self.mu_cons = params['mu_cons']

        # initializing trust region
        self.trust_region = params['trust_region']

        # Save the problem data
        self.tester.problem_data['Ntot'] = self.Ntot
        self.tester.problem_data['demand'] = self.demand
        self.tester.problem_data['capacity'] = self.Capacity
        self.tester.problem_data['rho'] = self.rho
        self.tester.problem_data['online_relationship_A'] = self.A
        self.tester.problem_data['online_relationship_B'] = self.B
        self.tester.problem_data['adj_mat'] = self.adj_mat
        self.tester.problem_data['weights'] = self.weight_mat

    #computing number of deaths with initial lockdown policy, which is L=1
    def compute_initial_death(self):

        self.s=0
        #going through the dynamics updates
        for t in range(self.n-1):
            for i in range(self.m):
                self.I[t+1][i]+=self.Ts*(-self.gamma*self.I[t][i]
                                         -self.Gamma_linear*self.I[t][i])
                self.D[t+1][i]=self.D[t][i]+self.Ts *self.Gamma_linear*self.I[t][i]
                self.R[t+1][i]=self.R[t][i]+self.Ts * self.gamma*self.I[t][i]

                for j in range(self.m):
                    Ntotval = 0
                    for l in range(self.m):
                        Ntotval = Ntotval + self.phi[t][l][j] * self.Ntot[l]

                    for k in range(self.m):
                        if ((i,j)) in self.adj_list and ((k,j)) in self.adj_list:

                            self.S[t+1][i]+=-self.rho[i]*self.beta*self.Ts*self.phi[t][i][j]*self.S[t][i]*self.phi[t][k][j]*self.I[t][k]/Ntotval
                            self.I[t+1][i]+=self.rho[i]*self.beta*self.Ts*self.phi[t][i][j]*self.S[t][i]*self.phi[t][k][j]*self.I[t][k]/Ntotval
                self.S[t+1][i]+=self.S[t][i]
                self.I[t+1][i]+=self.I[t][i]
        self.infect_cost=0

        #recording number of deaths
        for t in range(self.n-1):

            for i in range(self.m):
                self.s = self.s + (self.I[t][i])/1e2
                self.infect_cost=self.infect_cost+(self.I[t][i])/1e2

        #lockdown cost
        for t in range(self.n-1):
            for i in range(self.m):
                for l in range(self.num_entity):

                    self.s = self.s + (1-self.L[t][i][l])* (1-self.L[t][i][l])* self.econ_param*self.Ntot[i]
        self.infect_cost_best=self.infect_cost
        self.econ_cost_best=0
        #saving the best death so far
        self.sbest=self.s
        self.sbest=1e10
        print("initial death plus econ cost:",self.s)

        #saving best values to initialize the method
        self.I_best = self.I
        self.S_best = self.S
        self.D_best = self.D
        self.R_best = self.R
        self.phi_best = self.phi
        self.L_best = self.L
        

    #builds the main optimization Gurobi model once, for the future iterations
    def build_model(self):

        #initialize the variables
        self.I_var = [[self.gurobi_model.addVar(lb=0) for _ in range(self.m)] for zz in range(self.n)]
        self.D_var = [[self.gurobi_model.addVar(lb=0) for _ in range(self.m)] for zz in range(self.n)]
        self.R_var = [[self.gurobi_model.addVar(lb=0) for _ in range(self.m)] for zz in range(self.n)]
        self.S_var = [[self.gurobi_model.addVar(lb=0) for _ in range(self.m)] for zz in range(self.n)]
        self.L_var = [[[self.gurobi_model.addVar(lb=0, ub=1) for _ in range(self.num_entity)] for zz in range(self.m)]
                      for zzz in range(self.n)]
        self.o_var = [[[self.gurobi_model.addVar(lb=0) for _ in range(self.num_entity)] for zz in range(self.m)]
                      for zzz in range(self.n)]
        #initialize phi
        self.phi_var = [[[self.gurobi_model.addVar(lb=0, ub=1) for _ in range(self.m)] for zz in range(self.m)] for zzzz in range(self.n)]
        self.conv_var = [[[[0 for _ in range(self.m)] for _ in range(self.m)] for zz in range(self.m)] for zzzz in range(self.n)]

        self.f_var=[[[[self.gurobi_model.addVar(lb=0) for _ in range(self.num_entity)] for _ in range(self.m)]
                     for zz in range(self.m)] for zzzz in range(self.n)]

        #these next there variables are so-called slack variables for the "convexified" constraints
        self.tau_var_pos = [[[[0 for _ in range(self.m)] for _ in range(self.m)] for zz in range(self.m)] for zzzz in range(self.n)]

        self.tau_var_neg = [[[[0 for _ in range(self.m)] for _ in range(self.m)] for zz in range(self.m)] for zzzz in range(self.n)]

        self.tau_var_pos_phi = [[self.gurobi_model.addVar(lb=0) for zz in range(self.m)] for zzzz in range(self.n)]

        self.dyn_var_pos_R = [[self.gurobi_model.addVar(lb=0)  for _ in range(self.m)]  for zzzz in range(self.n)]

        self.dyn_var_neg_R = [[self.gurobi_model.addVar(lb=0)  for _ in range(self.m)] for zzzz in range(self.n)]

        self.dyn_var_pos_S = [[self.gurobi_model.addVar(lb=0)  for _ in range(self.m)]  for zzzz in range(self.n)]

        self.dyn_var_neg_S = [[self.gurobi_model.addVar(lb=0)  for _ in range(self.m)] for zzzz in range(self.n)]

        self.dyn_var_pos_I = [[self.gurobi_model.addVar(lb=0)  for _ in range(self.m)]  for zzzz in range(self.n)]

        self.dyn_var_neg_I = [[self.gurobi_model.addVar(lb=0)  for _ in range(self.m)] for zzzz in range(self.n)]


        #initialize the convexified variables for cities with adjacencies
        for t in range(self.n - 1):
            for i in range(self.m):

                for j in range(self.m):

                    for k in range(self.m):

                        if ((i,j)) in self.adj_list and ((k,j)) in self.adj_list:
                            self.tau_var_pos[t][i][j][k]=self.gurobi_model.addVar(lb=0)
                            self.tau_var_neg[t][i][j][k]=self.gurobi_model.addVar(lb=0)
                            self.conv_var[t][i][j][k]=self.gurobi_model.addVar(lb=0)


        #update the model
        self.gurobi_model.update()

    # adding convex constraints, check overleaf document
    def build_convex_constraints(self):

        # initial values of the variables, they are constraints (14g--h)
        for i in range(self.m):
            self.gurobi_model.addConstr(self.I_var[0][i] == self.Iinit[i] / self.scale_fac)
            self.gurobi_model.addConstr(self.S_var[0][i] == self.Sinit[i] / self.scale_fac)
            self.gurobi_model.addConstr(self.D_var[0][i] == self.Dinit[i] / self.scale_fac)
            self.gurobi_model.addConstr(self.R_var[0][i] == self.Rinit[i] / self.scale_fac)

        for t in range(self.n-1 ):
            for i in range(self.m):
                for l in range(self.num_entity):
                    # if l>0:
                    #     self.gurobi_model.addConstr(self.L_var[t][i][l] == self.L_var[t][i][l-1])
                    # if i>0:
                    #    self.gurobi_model.addConstr(self.L_var[t][i][l] == self.L_var[t][i-1][l])

                    if t>0 and not t % self.timer_val==0:
                        #print(t,i)
                        self.gurobi_model.addConstr(self.L_var[t][i][l] == self.L_var[t-1][i][l])


        #constraints on the semantics of phi
        for t in range(self.n - 1):
            for i in range(self.m):
                phi_cons = 0
                self.gurobi_model.addConstr(self.phi_var[t][i][i] >= self.min_phi_val)
                for j in range(self.m):
                    if ((i, j)) in self.adj_list:

                        phi_cons += self.phi_var[t][i][j]
                    else:
                        self.gurobi_model.addConstr(self.phi_var[t][i][j] == 0.0)
                #phi sums up to less than 1
                self.gurobi_model.addConstr( phi_cons<=1)

        #constraint(14i)
        for t in range(self.n - 1):
            for i in range(self.m):
                for l in range(self.num_entity):
                    self.gurobi_model.addConstr(self.o_var[t][i][l] == self.A[t][i][l]*self.L_var[t][i][l]+self.B[t][i][l])

        #constraints (14o)
        for t in range(self.n - 1):
            for i in range(self.m):
                phi_cons = 0

                for j in range(self.m):
                    phi_cons += self.phi_var[t][i][j]
                phi_cons2=0
                for l in range(self.num_entity):
                    phi_cons2 += (self.demand[i][l]-self.o_var[t][i][l])/self.Ntot[i]
                self.gurobi_model.addConstr(phi_cons==phi_cons2)


        #constraints(14n)
        for t in range(self.n - 1):
            for i in range(self.m):

                for j in range(self.m):
                    phi_cons2 = 0

                    for l in range(self.num_entity):
                        phi_cons2 += (self.f_var[t][i][j][l])
                    self.gurobi_model.addConstr(self.phi_var[t][i][j]==phi_cons2/self.Ntot[i])

        #constraints 14l and 14k
        for t in range(self.n - 1):
            for i in range(self.m):

                for j in range(self.m):
                    for l in range(self.num_entity):

                        if not i==j:
                            # constraints(14l)
                            self.gurobi_model.addConstr(self.f_var[t][i][j][l] >=
                                (self.demand[i][l] - self.o_var[t][i][l]-self.Capacity[i][l] * self.L_var[t][i][l])* self.weight[i][j][l] / self.Ntot[i])
                        else:
                            # constraints(14k)
                            self.gurobi_model.addConstr(self.f_var[t][i][j][l]<=self.Capacity[i][l] * self.L_var[t][i][l])
                            self.gurobi_model.addConstr(self.f_var[t][i][j][l]<=self.demand[i][l]-self.o_var[t][i][l])


        # constraints(14m)
        for t in range(self.n - 1):
            for l in range(self.num_entity):

                for j in range(self.m):
                    phi_cons = 0

                    for i in range(self.m):
                        phi_cons += self.f_var[t][i][j][l]

                    self.gurobi_model.addConstr(phi_cons <= self.Capacity[j][l] * self.L_var[t][j][l])



        #this loop constructs (14c--14f)
        for t in range(self.n - 1):
            for i in range(self.m):
                self.Ival = 0
                self.Sval = 0
                self.Ival += self.Ts * (- self.gamma * self.I_var[t][i]-self.Gamma_linear*self.I_var[t][i])
                self.gurobi_model.addConstr(self.D_var[t + 1][i] == self.D_var[t][i]
                                            +self.Ts * self.Gamma_linear*self.I_var[t][i])
                self.gurobi_model.addConstr(self.R_var[t + 1][i] == self.R_var[t][i] + self.Ts * self.gamma * self.I_var[t][i]
                                            -self.dyn_var_neg_R[t][i]+self.dyn_var_pos_R[t][i])

                for j in range(self.m):
                    Ntotval = 0
                    for l in range(self.m):
                        Ntotval = Ntotval + self.phi[t][l][j] * self.Ntot[l]
                    for k in range(self.m):
                        if ((i,j)) in self.adj_list and ((k,j)) in self.adj_list:

                            #conv_var refers to a convexified value of the variables in the 2nd and 3rd constraints
                        #on the equation(14c-14d) in the overleaf document
                            self.Sval += -self.beta * self.rho[i]*self.Ts * self.conv_var[t][i][j][k] / Ntotval
                            self.Ival += self.beta * self.rho[i] *self.Ts * self.conv_var[t][i][j][k] / Ntotval
                self.Sval += self.S_var[t][i]
                self.Ival += self.I_var[t][i]
                self.gurobi_model.addConstr(self.I_var[t + 1][i] == self.Ival-self.dyn_var_neg_I[t][i]+self.dyn_var_pos_I[t][i])
                self.gurobi_model.addConstr(self.S_var[t + 1][i] == self.Sval-self.dyn_var_neg_S[t][i]+self.dyn_var_pos_S[t][i])


    def build_nonconvex_constraints(self):
        #convexified version of the nonlinear functions of (14c)--(14d)
        for t in range(self.n - 1):
            for i in range(self.m):

                for j in range(self.m):

                    for k in range(self.m):
                        pass
                        if ((i,j)) in self.adj_list and ((k,j)) in self.adj_list:

                        #the following is a complete disasterous way to write these constraints, but couldnt find a better data structure
                                if abs( self.phi[t][i][j] * self.S[t][i] * self.phi[t][k][j] * self.I[t][k])>1e-6:
                                    self.remove_set.append(self.gurobi_model.addConstr(self.conv_var[t][i][j][k]==
                                    + self.phi[t][i][j] * self.S[t][i] * self.phi[t][k][j] * self.I_var[t][k]
                                    + self.phi[t][i][j] * self.S[t][i] * self.phi_var[t][k][j] * self.I[t][k]
                                    + self.phi[t][i][j] * self.S_var[t][i] * self.phi[t][k][j] * self.I[t][k]
                                    + self.phi_var[t][i][j] * self.S[t][i] * self.phi[t][k][j] * self.I[t][k]
                                    - 3 * self.phi[t][i][j] * self.S[t][i] * self.phi[t][k][j] * self.I[t][k]
                                    + self.tau_var_pos[t][i][j][k]-self.tau_var_neg[t][i][j][k]))
                                else:
                                    self.remove_set.append(self.gurobi_model.addConstr(self.conv_var[t][i][j][k]==
                                    + self.phi[t][i][j] * self.S[t][i] * self.phi[t][k][j] * self.I_var[t][k]
                                    + self.phi[t][i][j] * self.S[t][i] * self.phi_var[t][k][j] * self.I[t][k]
                                    + self.phi[t][i][j] * self.S_var[t][i] * self.phi[t][k][j] * self.I[t][k]
                                    + self.phi_var[t][i][j] * self.S[t][i] * self.phi[t][k][j] * self.I[t][k]
                                    + self.tau_var_pos[t][i][j][k]-self.tau_var_neg[t][i][j][k]))

        #adding trust region constraints for the phi variables
        for t in range(self.n-1):
            for i in range(self.m):
                    #this if is just there for the first iteration, where I initialize things
                    if self.trust_region<10:
                        if self.phi[t][i][j]>1e-3:
                            self.remove_set.append(self.gurobi_model.addConstr(self.phi_var[t][i][j] <= self.phi[t][i][j] * self.trust_region))
                            self.remove_set.append(self.gurobi_model.addConstr(self.phi_var[t][i][j] >= self.phi[t][i][j] / self.trust_region))
                        else:
                            self.remove_set.append(self.gurobi_model.addConstr(self.phi_var[t][i][j] <= 1e-3))


    def set_objective(self):
        # setting up objective, these objectives are related to violations of the convexified constraints on the equation (11) in the overleaf document
        self.obj = 0
        for t in range(self.n - 1):
            for i in range(self.m):

                self.obj = self.obj + self.tau_var_pos_phi[t][i] * self.mu_cons

                self.obj=self.obj+ self.dyn_var_pos_S[t][i]*self.mu_cons
                self.obj=self.obj+ self.dyn_var_neg_S[t][i]*self.mu_cons

                self.obj=self.obj+ self.dyn_var_pos_I[t][i]*self.mu_cons
                self.obj=self.obj+ self.dyn_var_neg_I[t][i]*self.mu_cons

                self.obj=self.obj+ self.dyn_var_pos_R[t][i]*self.mu_cons
                self.obj=self.obj+ self.dyn_var_neg_R[t][i]*self.mu_cons

                for j in range(self.m):
                    for k in range(self.m):
                        if ((i,j)) in self.adj_list and ((k,j)) in self.adj_list:

                            self.obj = self.obj + self.tau_var_pos[t][i][j][k] * self.mu_cons
                            self.obj = self.obj + self.tau_var_neg[t][i][j][k] * self.mu_cons

                #these are the actual objectives, the above values are for slack variables
                self.obj = self.obj + (self.I_var[t][i])/1e2

                for l in range(self.num_entity):
                    self.obj = self.obj + (1-self.L_var[t][i][l])  *(1- self.L_var[t][i][l]) * self.econ_param *self.Ntot[i]

    def solve_problem(self):
        #update gurobi parameters
        self.gurobi_model.update()
        self.gurobi_model.setParam('OutputFlag', 1)
        self.gurobi_model.setParam('NumericFocus', 3)
        self.gurobi_model.Params.BarHomogeneous = 1.0
        # self.gurobi_model.Params.ScaleFlag = 2
        self.gurobi_model.Params.Method = 2
        self.gurobi_model.Params.Crossover = 0
        self.gurobi_model.Params.NumericFocus = 3
        self.gurobi_model.Params.FeasibilityTol = 1e-6
        self.gurobi_model.Params.OptimalityTol = 1e-6
        self.gurobi_model.Params.BarConvTol = 1e-6

        # solve the problem
        self.gurobi_model.setObjective(self.obj, GRB.MINIMIZE)
        print('Solving...')
        self.gurobi_model.optimize()

        self.L_aux = [[[0 for _ in range(self.num_entity)]for _ in range(self.m)] for zz in range(self.n)]
        self.phi_aux = [[[0 for _ in range(self.m)] for zz in range(self.m)] for zzzz in range(self.n)]

        # initialize L and phi for the values of the optimization problem
        for t in range(self.n - 1):
            for i in range(self.m):
                for l in range(self.num_entity):
                    self.L_aux[t][i][l] = self.L_var[t][i][l].x

        for t in range(self.n - 1):
            for i in range(self.m):
                for j in range(self.m):
                    self.phi_aux[t][i][j] = self.phi_var[t][i][j].x
        #print(self.phi_aux)


    #check if the solution improves the previous value
    def check_solution(self):
        self.sval = 0

        print("checking results")

        # defining auxillary variables for computing a feasible phi for a given L
        self.I_aux = np.zeros((self.n, self.m))
        self.S_aux = np.zeros((self.n, self.m))
        self.R_aux = np.zeros((self.n, self.m))
        self.D_aux = np.zeros((self.n, self.m))

        # scaling down the initial condition
        self.I_aux[0][:] = self.Iinit / self.scale_fac
        self.R_aux[0][:] = self.Rinit / self.scale_fac
        self.D_aux[0][:] = self.Dinit / self.scale_fac
        self.S_aux[0][:] = self.Sinit / self.scale_fac


        #dynamics update for the variables
        for t in range(self.n - 1):
            for i in range(self.m):
                self.I_aux[t+1][i]+=self.Ts*(-self.gamma*self.I_aux[t][i]-self.Gamma_linear*self.I_aux[t][i])
                self.D_aux[t+1][i]=self.D_aux[t][i]+self.Ts *self.Gamma_linear*self.I_aux[t][i]
                self.R_aux[t + 1][i] = self.R_aux[t][i] + self.Ts * self.gamma * self.I_aux[t][i]


                for j in range(self.m):
                    Ntotval = 0
                    for l in range(self.m):
                        Ntotval = Ntotval + self.phi_aux[t][l][j] * self.Ntot[l]
                    for k in range(self.m):
                        if ((i,j)) in self.adj_list and ((k,j)) in self.adj_list:

                            self.S_aux[t + 1][i] += -self.rho[i]*self.beta * self.Ts * \
                                self.phi_aux[t][i][j] * self.S_aux[t][i] * self.phi_aux[t][k][j] * self.I_aux[t][k] / Ntotval
                            self.I_aux[t + 1][i] +=  self.rho[i]*self.beta * self.Ts * \
                                self.phi_aux[t][i][j] * self.S_aux[t][i] * self.phi_aux[t][k][j] * self.I_aux[t][k] / Ntotval

                self.S_aux[t + 1][i] += self.S_aux[t][i]
                self.I_aux[t + 1][i] += self.I_aux[t][i]

        #cost update
        self.infect_cost=0
        for t in range(self.n-1):

            for i in range(self.m):
                self.sval = self.sval + (self.I_aux[t][i])/1e2
                self.infect_cost=self.infect_cost+(self.I_aux[t][i])/1e2
        self.econ_cost=0
        for t in range(self.n-1):
            for i in range(self.m):
                for l in range(self.num_entity):

                    self.sval = self.sval + (1-self.L_aux[t][i][l])*\
                                (1-self.L_aux[t][i][l])* self.econ_param *self.Ntot[i]
                    self.econ_cost=self.econ_cost+ (1-self.L_aux[t][i][l])*\
                                (1-self.L_aux[t][i][l])* self.econ_param*self.Ntot[i]

        print(self.sval, self.sbest)
        print(self.Ntot)
        # if new death count is better than best
        if self.sval < self.sbest:
            #print(self.phi)
            print("better")

            #update the initial condition and best values
            self.I = self.I_aux
            self.S = self.S_aux
            self.D = self.D_aux
            self.R = self.R_aux
            self.L = self.L_aux
            self.phi = self.phi_aux
            #print(self.phi)
            self.I_best = self.I_aux
            self.S_best = self.S_aux
            self.D_best = self.D_aux
            self.R_best = self.R_aux
            self.phi_best = self.phi_aux
            self.L_best = self.L_aux
            self.infect_cost_best=self.infect_cost
            self.econ_cost_best=self.econ_cost_best

            #increase the size of trust region
            self.trust_region = (min(1.5, (self.trust_region - 1) * 1.2 + 1))
            #keep the best count
            self.sbest = self.sval
        #otherwise
        else:
            #do not update, shrink trust rebion
            self.trust_region = ((self.trust_region - 1) / 1.2 + 1)

        print("trust region", self.trust_region)
        print("best objective val", self.sbest)


    def save_solution(self):
        self.tester.results['I_best'] = self.I_best
        self.tester.results['S_best'] = self.S_best
        self.tester.results['D_best'] = self.D_best
        self.tester.results['R_best'] = self.R_best
        self.tester.results['phi_best'] = self.phi_best
        self.tester.results['L_best'] = self.L_best
        self.tester.results['econ_cost']=self.econ_cost_best
        self.tester.results['infect_cost']=self.infect_cost_best

        with open(os.path.join(self.tester.params['save_file_path'], self.tester.results['time_cur_str']), 'wb') as fp:
            pickle.dump(self.tester, fp)

    #get best values and plot the result
    def plot_solution(self):
        Ivalue = []
        for t in range(self.n):
            Iappend = np.zeros((1))

            for i in range(self.m):
                Iappend+=self.I_best[t][i]
            Ivalue.append(Iappend)

        Svalue = []
        for t in range(self.n):
            Sappend = np.zeros((1))

            for i in range(self.m):
                Sappend+=self.S_best[t][i]
            Svalue.append(Sappend)

        Rvalue = []
        for t in range(self.n):
            Rappend = np.zeros((1))

            for i in range(self.m):
                Rappend+=self.R_best[t][i]
            Rvalue.append(Rappend)

        Dvalue = []
        for t in range(self.n):
            Dappend = np.zeros((1))

            for i in range(self.m):
                Dappend+=self.D_best[t][i]
            Dvalue.append(Dappend)


        time = range(self.n)
        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 9)

        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.plot(time, Ivalue, 'r-', label='infectious', linewidth=3.0)
        plt.plot(time, Dvalue, 'm-', label='dead', linewidth=3.0)
        plt.plot(time, Svalue, 'b-', label='susceptible', linewidth=3.0)
        plt.plot(time, Rvalue, 'g-', label='immune', linewidth=3.0)
        plt.xlabel('time (days)', fontsize=18)
        plt.ylabel('number of individuals (K)', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(0, 100)
        #plt.tight_layout()

        savename = os.path.join(self.tester.params['save_file_path'], 'cases_plots', 'cases_plot_' + self.tester.results['time_cur_str'] + '.png')
        plt.savefig(savename, dpi = 300)
        plt.show()
        Lvalue = [[] for _ in range(self.m)]
        #print(self.L_best)
        for t in range(self.n):
            for i in range(self.m):
                Iappend = np.zeros((1))

                for l in range(self.num_entity):

                    Iappend+=self.L_best[t][i][l]/(self.num_entity)
                Lvalue[i].append(Iappend)


        plt.clf()
        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 9)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        # for i in range(self.m):
        #     plt.plot(time, Lvalue[i], label='Capacity Ratio for City '+str(i), linewidth=3.0)
        for i in range(math.floor((self.m/2))):
            plt.plot(time, Lvalue[i], label='Capacity Ratio for City '+str(i+1), linewidth=3.0)
        plt.xlabel('time (days)', fontsize=18)
        plt.ylabel('Capacity Ratio', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(0, self.n-2)
        #plt.tight_layout()
        savename = os.path.join(self.tester.params['save_file_path'], 'capacity_plots', 'capacity_' + self.tester.results['time_cur_str'] + '.png')

        plt.savefig(savename, dpi = 300)
        plt.show()
        plt.clf()
        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(16, 9)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        for i in range(math.floor((self.m/2)),self.m):
            plt.plot(time, Lvalue[i], label='Capacity Ratio for City '+str(i+1), linewidth=3.0)
        plt.xlabel('time (days)', fontsize=18)
        plt.ylabel('Capacity Ratio', fontsize=18)
        plt.legend(fontsize=18)
        plt.xlim(0, self.n-2)
        #plt.tight_layout()
        savename = os.path.join(self.tester.params['save_file_path'], 'capacity_plots', 'capacity_' + self.tester.results['time_cur_str'] + '_l' + '.png')

        plt.savefig(savename, dpi = 300)
        plt.show()

    #main function for running the code
    def run(self):
        #compute initial death
        self.compute_initial_death()
        for iter in range(self.num_iter):
            print("encoding")
            print("iternumber",iter)
            #set for certain constraints that we iteratively build
            self.remove_set = []
            #build the variables and some of the constraints for the first iteration
            if iter==0:
                # build model once
                self.gurobi_model=Model("qcp")
                self.gurobi_model.update()
                self.build_model()

                # build convex constraints once
                self.build_convex_constraints()

            #build nonconvex constraints
            self.build_nonconvex_constraints()

            #update model and set objective
            self.gurobi_model.update()
            self.set_objective()

            #solver parameters
            self.solve_problem()

            #check if the solution is better
            self.check_solution()

            #break if the trust region is very small
            if self.trust_region<1+1e-2:
                break
            #remove convexified constraints
            self.gurobi_model.remove(self.remove_set)
            #set remove set to empty after removing convexified constraints
            self.remove_set=[]

        time_cur = datetime.datetime.now()
        time_cur_str = r'{}'.format(time_cur.strftime("%Y-%m-%d_%H-%M-%S"))
        self.tester.results['time_cur_str'] = time_cur_str

        #saver function
        self.save_solution()
        #plotter function
        self.plot_solution()

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from params_config import params
    from utils.tester import Tester

    tester = Tester()
    tester.set_params(params)

    solver = gurobi_solver(tester)
    solver.run()
