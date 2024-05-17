import numpy as np
import math
import time
import Algorithms as alg
import csv
import gurobipy as gp
from gurobipy import GRB
import os

tabu_list = ["AGG.mps"]
# cannot be opened: AGG.mps

folderNETLIB = "NETLIB_Instances\\"
folderCESetup = "InstancesCESetup\\"

list_files = os.listdir(folderNETLIB)

jobNr = 5
num_columns = [5] #1,5,or 10
numInstances = 20

for file in list_files:
    if file in tabu_list:
        pass
    else:
        print(file)
        instance_file = folderNETLIB+file
    
        #read and solve present problem
        start = time.time() 
        n,m,c,A,b,sense_constrs,lb_vars,ub_vars,model_sense, x_opt, opt = alg.getProblemParameters(instance_file)
        end = time.time()
        runTime = end-start
        
        start = time.time() 
        opt_myModel, x_opt, setupTime = alg.solvePresentProblem(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,model_sense)
        end = time.time()
        runTime = end-start
        
        csvPresentProblems = open("PresentProblem_Output.csv", 'a')
        out = csv.writer(csvPresentProblems,delimiter=";", lineterminator = "\n")
        row = [file,n,m,opt_myModel,runTime,setupTime]
        out.writerow(row)
        csvPresentProblems.close()

        
        for instanceID in range(numInstances):
            #Get Setup
            for numColumns in num_columns:
                list_vars, list_bounds, mutable_c,perc_c,mutable_A, perc_A, mutable_b,perc_b = alg.getCESetup(folderCESetup + file.replace(".mps","") + "_" +str(instanceID) + "_" + str(numColumns))
                
                ############################
                ###Relative Counterfactuals
                ##########################
                start = time.time() 
                print("Start Calculations Relative Counterfactuals.")
                feasible_rel,no_solution_found_rel,hit_timelimit_rel,nonZero_c_rel,nonZero_b_rel,nonZero_A_rel,c_rCE,A_rCE,b_rCE,x_rCE,opt_dist_rel = \
                    alg.getRelativeCounterfactual(n,m,c,A,b,opt,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b,1)
                end = time.time()
                runTime = end-start
                
                csvRelativeCEProblem = open("RelativeCEProblem_Output_"+ str(jobNr) + ".csv", 'a')
                out = csv.writer(csvRelativeCEProblem,delimiter=";", lineterminator = "\n")
                if feasible_rel and not no_solution_found_rel:
                    row = [file,instanceID,n,m,numColumns,feasible_rel,no_solution_found_rel,hit_timelimit_rel,nonZero_c_rel,nonZero_b_rel,nonZero_A_rel,np.round(opt_dist_rel,2),np.round(np.dot(c,x_opt),2),np.round(np.dot(c,x_rCE),2),np.round(np.dot(c_rCE,x_opt),2),np.round(np.dot(c_rCE,x_rCE),2),len(list_vars),len(mutable_c), len(mutable_A), len(mutable_b), np.round(runTime,2)]
                else: 
                    row = [file,instanceID,n,m,numColumns,feasible_rel,no_solution_found_rel,hit_timelimit_rel,0,0,0,np.round(opt_dist_rel,2),0,0,0,0,len(list_vars),len(mutable_c), len(mutable_A), len(mutable_b), np.round(runTime,2)]
                out.writerow(row)
                csvRelativeCEProblem.close()
                

                start = time.time() 
                print("Start Calculations Relative Counterfactuals Linearization.")
                feasible_rel,no_solution_found_rel,hit_timelimit_rel,nonZero_c_rel,nonZero_b_rel,nonZero_A_rel,c_rCE,A_rCE,b_rCE,x_rCE,opt_dist_rel = \
                    alg.getRelativeCounterfactualLinearization(n,m,c,A,b,opt,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b,1)
                end = time.time()
                runTime = end-start
                
                csvRelativeCEProblem = open("RelativeCEProblem_Output_Linearization_" + str(jobNr) + ".csv", 'a')
                out = csv.writer(csvRelativeCEProblem,delimiter=";", lineterminator = "\n")
                if feasible_rel and not no_solution_found_rel:
                    row = [file,instanceID,n,m,numColumns,feasible_rel,no_solution_found_rel,hit_timelimit_rel,nonZero_c_rel,nonZero_b_rel,nonZero_A_rel,np.round(opt_dist_rel,2),np.round(np.dot(c,x_opt),2),np.round(np.dot(c,x_rCE),2),np.round(np.dot(c_rCE,x_opt),2),np.round(np.dot(c_rCE,x_rCE),2),len(list_vars),len(mutable_c), len(mutable_A), len(mutable_b), np.round(runTime,2)]
                else: 
                    row = [file,instanceID,n,m,numColumns,feasible_rel,no_solution_found_rel,hit_timelimit_rel,0,0,0,np.round(opt_dist_rel,2),0,0,0,0,len(list_vars),len(mutable_c), len(mutable_A), len(mutable_b), np.round(runTime,2)]
                out.writerow(row)
                csvRelativeCEProblem.close()
                

        




    