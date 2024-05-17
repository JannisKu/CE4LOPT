import numpy as np
import math
import time
import Algorithms as alg
import csv
import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd
import scipy.sparse as sp


def printSolution(x_opt):
    for s in range(len(S)):
        print(S[s] + ":",end="")
        for j in range(len(N)):
            if x_opt[s*len(N) + j]>0.0001:
                print(" ",N[j] + ":",x_opt[s*len(N) + j], end="")
        print("\n")
    

#setup present problem
nutr_val = pd.read_excel('Syria_instance_edited.xlsx', sheet_name='nutr_val', index_col='Food')
nutr_req = pd.read_excel('Syria_instance_edited.xlsx', sheet_name='nutr_req', index_col='Type')
cost_p = pd.read_excel('Syria_instance_edited.xlsx', sheet_name='FoodCost', index_col='Supplier')

CEType = "strong"

print("CEType:",CEType)

N = list(nutr_val.index)  # foods
M = nutr_req.columns  # nutrient requirements
S = list(cost_p.index)

#N = [supplier_index*len(N) + np.where(np.array(N)=='Beans')[0][0]]

n = len(S)*len(N)
m = len(M)+2

c = np.zeros(n)
for s in range(len(S)):
    for i in range(len(N)):
        c[s*len(N) + i] = cost_p.iloc[s,i]
        
A = np.zeros((m,n))
b = np.zeros(m)

sense_constrs = []
lb_vars = np.zeros(n)
ub_vars = np.ones(n)*100

for i in range(len(M)):
    b[i] = nutr_req.iloc[0,i]
    sense_constrs.append(">")
    for s in range(len(S)):
        for j in range(len(N)):
            A[i,s*len(N) + j] = nutr_val.iloc[j,i]
            
for s in range(len(S)):
    for j in range(len(N)):
        if j==20:
            A[len(M),s*len(N) + j] = 1
        elif j==9:
            A[len(M)+1,s*len(N) + j] = 1
            
b[len(M)] = 0.2
b[len(M)+1] = 0.05
sense_constrs.append("=")
sense_constrs.append("=")

A = sp.csr_matrix(A[:,:])

#Solve present problem
opt, x_opt, setuptime = alg.solvePresentProblem(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,1)

printSolution(x_opt)
print(opt)



#Define counterfactual setup

for supplier_index in range(5):

    list_vars = [supplier_index*len(N) + np.where(np.array(N)=='Beans')[0][0], \
                 supplier_index*len(N) + np.where(np.array(N)=='Wheat')[0][0]]
    list_bounds = [(1.0,100),(2.5,100)]
    
    print(list_vars)
    print(list_bounds)
    
    #mutable cost-parameters
    mutable_c = list(np.arange(supplier_index*len(N),(supplier_index+1)*len(N)))
    
    perc_c = 1
    mutable_A = []
    for i in range(len(M)):
        for j in range(supplier_index*len(N),(supplier_index+1)*len(N)):
            mutable_A.append((i,j))
    
    mutable_b = []
    perc_A = 1
    perc_b = 1
    
    if CEType == "weak":
        #Calculate Weak CEs
        start = time.time()
        feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_CE,A_CE,b_CE,x_CE,opt_dist = alg.getWeakCounterfactual(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b, 0, [])
        end = time.time()
        runTime = end-start
    elif CEType == "strong":
        #Calculate Strong CEs
        start = time.time()
        feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_CE,A_CE,b_CE,x_CE,opt_dist = alg.getStrongCounterfactual(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b)
        end = time.time()
        runTime = end-start
    elif CEType == "relative":
        #Calculate Relative CEs
        start = time.time()
        feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_CE,A_CE,b_CE,x_CE,opt_dist = alg.getRelativeCounterfactual(n,m,c,sp.csr_matrix(A),b,opt,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b)
        end = time.time()
        runTime = end-start
    
    #Output
    print(feasible,no_solution_found,hit_timelimit)
    
    if not no_solution_found:
        opt_CE, x_opt_CE, setuptime = alg.solvePresentProblem(n,m,c_CE,A_CE,b_CE,sense_constrs,lb_vars,ub_vars,1)
        opt_CE_Dx, x_opt_CE_Dx = alg.solvePresentProblemPlusDx(n,m,c_CE,A_CE,b_CE,sense_constrs,lb_vars,ub_vars,1,list_vars,list_bounds)
        printSolution(x_opt_CE)
        
        csvFile = open("Diet_Problem_PrizesCEs_" + CEType + ".csv", 'a')
        out = csv.writer(csvFile,delimiter=";", lineterminator = "\n")
        row = [supplier_index,x_opt_CE[list_vars[0]],x_opt_CE[list_vars[1]]]
        
        print("Bought Beans:",x_opt_CE[list_vars[0]])
        print("Bought Wheat:",x_opt_CE[list_vars[1]])
        
        print("Number cost parameters to change:",nonZero_c)
        for j in range(n):
            if abs(c_CE[j]-c[j])>0.00001:
                print("costs",N[j-supplier_index*len(N)],":",c[j],"-->",c_CE[j])
                row.append(N[j-supplier_index*len(N)])
                row.append(c[j])
                row.append(c_CE[j])
                
        print("Number nutrition parameters to change:",nonZero_A)
        for i in range(m):
            for j in range(n):
                if abs(A_CE[i,j]-A[i,j])>0.00001:
                    print("nutrition value",M[i],"in",N[j-supplier_index*len(N)],":",A[i,j],"-->",A_CE[i,j])
                    row.append(N[j-supplier_index*len(N)])
                    row.append(M[i])
                    row.append(A[i,j])
                    row.append(A_CE[i,j])
        
        print("Runtime:",runTime)
        row.append(runTime)
        
        out.writerow(row)
        csvFile.close()
    
    
    
    