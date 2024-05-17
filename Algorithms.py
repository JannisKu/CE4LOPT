import numpy as np
import math
import time
import gurobipy as gp
from gurobipy import GRB
import copy
import scipy.sparse as sp

def printStatus(stat):
    
    if stat  == GRB.OPTIMAL:
        t=1
        #print('Model is optimal')
    elif stat  == GRB.INF_OR_UNBD:
        print('Model  is  infeasible  or  unbounded')
        #sys.exit()
    elif stat  == GRB.INFEASIBLE:
        print('Model  is  infeasible')
        #sys.exit()
    elif stat  == GRB.UNBOUNDED:
        print('Model  is  unbounded')
        #sys.exit()
    else:
        print('Optimization  ended  with  status ' + str(stat))
        #sys.exit()
        
def readAndSolve(file):
    lp = gp.read(instanceFile)
    lp.setParam("OutputFlag",1)
    lp.setParam("TimeLimit",7200)

    lp.optimize()


def solvePresentProblem(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,model_sense):
    lp = gp.Model("Knapsack")
    lp.setParam("OutputFlag",0)
    lp.setParam("FeasibilityTol",0.000000001)
    lp.setParam("OptimalityTol",0.000000001)
    lp.setParam("TimeLimit",7200)
    
    start = time.time()
    lp.ModelSense = model_sense
    x = lp.addVars(n,obj=c,vtype=GRB.CONTINUOUS,lb=lb_vars,ub=ub_vars, name="x")
    
    for i in range(A.shape[0]):
        lhs = 0
        curr_row = A.getrow(i)
        for j in curr_row.indices:
            lhs = lhs + A[i,j]*x[j]
        lp.addConstr(lhs, sense=sense_constrs[i], rhs=b[i])
        
    end = time.time()
    
    setupTime = end-start
    
    lp.optimize()
    lp.write("PresentProblem.lp")
    
    return lp.objVal, np.array(list(lp.getAttr("X",x).values())), setupTime


def solvePresentProblemPlusDx(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,model_sense,list_vars,list_bounds):
    lp = gp.Model("Knapsack")
    lp.setParam("OutputFlag",0)
    lp.setParam("FeasibilityTol",0.000000001)
    lp.setParam("OptimalityTol",0.000000001)
    lp.setParam("TimeLimit",7200)
    
    lp.ModelSense = model_sense
    x = lp.addVars(n,obj=c,vtype=GRB.CONTINUOUS,lb=lb_vars,ub=ub_vars, name="x")
    
    
    for i in range(A.shape[0]):
        lhs = 0
        for j in range(n):
            lhs = lhs + A[i,j]*x[j]
        lp.addConstr(lhs, sense=sense_constrs[i], rhs=b[i])
        
        
    counter = -1
    for j in list_vars:
        counter+=1
        if math.isinf(list_bounds[counter][1]):
            pass
        else:
            lp.addConstr(1*x[j], sense="<", rhs=list_bounds[counter][1])
            
        if math.isinf(list_bounds[counter][0]):
            pass
        else:
            lp.addConstr(1*x[j], sense=">", rhs=list_bounds[counter][0])
        
    lp.optimize()
    lp.write("PresentProblem.lp")
    
    return lp.objVal, np.array(list(lp.getAttr("X",x).values()))
        
def getProblemParameters(instanceFile):
    lp = gp.read(instanceFile)
    lp.setParam("OutputFlag",0)
    lp.setParam("TimeLimit",7200)

    lp.optimize()

    n = lp.numVars
    m=lp.NumConstrs
    
    A = lp.getA()
    b = lp.getAttr("RHS",lp.getConstrs())
    c = lp.getAttr("Obj",lp.getVars())
    sense_constrs = lp.getAttr("Sense",lp.getConstrs())
    lb_vars = lp.getAttr("LB",lp.getVars())
    ub_vars = lp.getAttr("UB",lp.getVars())
    model_sense = lp.ModelSense #1= Minimization, -1=Maximization
    x_opt = lp.getAttr("X",lp.getVars())
    opt = lp.ObjVal
    
            
    return n,m,c,A,b,sense_constrs,lb_vars,ub_vars,model_sense, x_opt, opt

def generateCESetup(n,m,x_opt,lb,ub,c,A,b,list_num_columns,list_num_rows,instance_name,mutableType="random_columns"):
    for instanceID in range(20):
        start = time.time()
        now = time.time()
        countVars = 0
        list_vars = []
        list_bounds = []
        mutable_c = []
        mutable_A = []
        mutable_b = []
        
        positive_columns = np.where(np.array(lb)>=0)[0]
        
        while countVars < 3 and now-start<600:
            i = np.random.choice(np.arange(0,n),1)[0]
            if i not in list_vars:
                if ub[i]-x_opt[i]>0.0001:
                    if x_opt[i]>0.00001:
                        list_bounds.append((min(x_opt[i]*1.05,ub[i]),ub[i]))
                    elif x_opt[i]<-0.00001:
                        list_bounds.append((min(x_opt[i]*0.95,ub[i]),ub[i]))
                    else:
                        list_bounds.append((min(ub[i],0.05),ub[i]))
                        
                    countVars+=1
                    list_vars.append(i)
                elif x_opt[i]-lb[i] > 0.0001:
                    if x_opt[i]>0.00001:
                        list_bounds.append((lb[i],max(x_opt[i]*0.95,lb[i])))
                    elif x_opt[i]<-0.00001:
                        list_bounds.append((lb[i],max(x_opt[i]*1.05,lb[i])))
                    else:
                        list_bounds.append((lb[i],max(-0.05, lb[i])))
                        
                    countVars+=1
                    list_vars.append(i)
                else:
                    pass
            now = time.time()
        
        if mutableType == "random_columns":
            for num_columns in list_num_columns:
                noParametersSelected = True
                start = time.time()
                now = time.time()
                while noParametersSelected and now-start<600:
                    list_columns = np.random.choice(positive_columns,num_columns, False)
                    for j in list_columns:
                        if "D6CUBE" in instance_name or "QAP" in instance_name or "SCTAP" in instance_name \
                            or "SHELL" in instance_name:
                            if abs(c[j])>10 and abs(10*np.round(c[j]/10.0,0)-c[j])>0.0001:
                                mutable_c.append(j)
                                
                            for i in range(m):
                                if abs(A[i,j])>10 and abs(10*np.round(A[i,j]/10.0,0)-A[i,j])>0.0001:
                                    mutable_A.append((i,j))
                        else:
                            if abs(np.round(c[j],0)-c[j])>0.0001:
                                mutable_c.append(j)
                                
                            for i in range(m):
                                if abs(np.round(A[i,j],0)-A[i,j])>0.0001:
                                    mutable_A.append((i,j))
                                
                    if len(mutable_c)>0 or len(mutable_A)>0:
                        noParametersSelected = False
                    now = time.time()
                    
                if len(mutable_c)==0 and len(mutable_A)==0 and len(mutable_b)==0:
                    print("NO MUTABLE PARAMETERS FOUND!")
                            
                f = open("InstancesCESetup\\" + instance_name + "_" + str(instanceID) + "_" + str(num_columns), "w")
                f.write("Variables\n")
                for val in list_vars:
                    f.write(str(val)+"\n")
                f.write("Bounds\n")
                for val in list_bounds:
                    f.write(str(val)+"\n")
                f.write("Mutable Cost Indices\n")
                for val in mutable_c:
                    f.write(str(val)+"\n")
                f.write("Mutable Constraint Parameters\n")
                for val in mutable_A:
                    f.write(str(val)+"\n")
                f.write("Mutable Rhs Parameters\n")
                for val in mutable_b:
                    f.write(str(val)+"\n")
                f.close() 
        

def getCESetup(file_name):
    list_vars = []
    list_bounds = []
    mutable_c = []
    mutable_A = []
    mutable_b = []
    
    perc_c = 1
    perc_A = 1
    perc_b = 1
    
    file = open(file_name,'r')
    
    for line in file.readlines():
        if "Variables" in line:
            block = 1
        elif "Bounds" in line:
            block = 2
        elif "Mutable Cost Indices" in line:
            block = 3
        elif "Mutable Constraint Parameters" in line:
            block = 4
        elif "Mutable Rhs Parameters" in line:
            block = 5
        else:
            if block == 1:
                list_vars.append(int(line))
            elif block == 2:
                lb = float(line.replace("(","").replace(")","").split(",")[0])
                ub = float(line.replace("(","").replace(")","").split(",")[1].replace(" ",""))
                list_bounds.append((lb,ub))
            elif block == 3:
                mutable_c.append(int(line))
            elif block == 4:
                row = int(line.replace("(","").replace(")","").split(",")[0])
                col = int(line.replace("(","").replace(")","").split(",")[1].replace(" ",""))
                mutable_A.append((row,col))
            elif block == 5:
                mutable_b.append(int(line))
    
    return list_vars, list_bounds, mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b

def getStrongCounterfactual(n,m,c_hat,A_hat,b_hat,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b):
    numLEq = 0
    numGEq = 0
    numEq = 0
    
    for sense in sense_constrs:
        if sense==">":
            numGEq+=1
        elif sense=="<":
            numLEq+=1
        elif sense=="=":
            numEq+=1
            
    n_c = len(mutable_c) 
    c_lb = np.zeros(n_c)
    c_ub = np.zeros(n_c)
    for j in range(n_c):
        if c_hat[mutable_c[j]]>=0:
            c_lb[j] = c_hat[mutable_c[j]]*(1-perc_c)
            c_ub[j] = c_hat[mutable_c[j]]*(1+perc_c)
        else:
            c_lb[j] = c_hat[mutable_c[j]]*(1+perc_c)
            c_ub[j] = c_hat[mutable_c[j]]*(1-perc_c)
    
    n_b = len(mutable_b)
    b_lb = np.zeros(n_b)
    b_ub = np.zeros(n_b)
    for i in range(n_b):
        if b_hat[mutable_b[i]]>=0:
            b_lb[i] = b_hat[mutable_b[i]]*(1-perc_b)
            b_ub[i] = b_hat[mutable_b[i]]*(1+perc_b)
        else:
            b_lb[i] = b_hat[mutable_b[i]]*(1+perc_b)
            b_ub[i] = b_hat[mutable_b[i]]*(1-perc_b)
    
    n_A = len(mutable_A)
    A_lb = np.zeros(n_A)
    A_ub = np.zeros(n_A)
    for i in range(n_A):
        if A_hat[mutable_A[i][0],mutable_A[i][1]]>=0:
            A_lb[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1-perc_A)
            A_ub[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1+perc_A)
        else:
            A_lb[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1+perc_A)
            A_ub[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1-perc_A)
    
    lp = gp.Model("StrongCE")
    lp.setParam("OutputFlag",1)
    lp.setParam("NonConvex",2)
    lp.setParam("FeasibilityTol",0.00000001)
    lp.setParam("OptimalityTol",0.00000001)
    #lp.setParam("Presolve",0)
    lp.setParam("TimeLimit",7200)
     
    lp.ModelSense=1
    
    #Add Variables
    
    L = len(list_vars)
    
    gamma = lp.addVars(L,ub=0.0, lb=-GRB.INFINITY, name="gamma")
    wGEq = lp.addVars(L,numGEq,name="w>")
    wLEq = lp.addVars(L,numLEq,lb=-GRB.INFINITY,ub=0,name="w<")
    wEq = lp.addVars(L,numEq,lb=-GRB.INFINITY,name="w=")
    wu = lp.addVars(L,n,lb=-GRB.INFINITY,ub=0,name="wu")
    wl = lp.addVars(L,n,name="wl")
    q = lp.addVars(L,n,lb=-GRB.INFINITY,name="q")
    
    c = lp.addVars(n_c,lb=c_lb,ub=c_ub,name="c")
    A = lp.addVars(n_A, lb=A_lb, ub=A_ub, name="A")
    b = lp.addVars(n_b, lb=b_lb, ub=b_ub, name="b")
    
    mu_c = lp.addVars(n_c, obj = 1, name="mu_c")
    mu_b = lp.addVars(n_b, obj = 1, name="mu_b")
    mu_A = lp.addVars(n_A, obj = 1, name="mu_A")
    
    for j in range(n_c):
        lp.addConstr(mu_c[j]+c[j], sense = ">", rhs=c_hat[mutable_c[j]])
        lp.addConstr(mu_c[j]-c[j], sense = ">", rhs=-c_hat[mutable_c[j]])
        
    for i in range(n_b):
        lp.addConstr(mu_b[i]+b[i], sense = ">", rhs=b_hat[mutable_b[i]])
        lp.addConstr(mu_b[i]-b[i], sense = ">", rhs=-b_hat[mutable_b[i]])
        
    for i in range(n_A):
        lp.addConstr(mu_A[i]+A[i], sense = ">", rhs=A_hat[mutable_A[i][0],mutable_A[i][1]])
        lp.addConstr(mu_A[i]-A[i], sense = ">", rhs=-A_hat[mutable_A[i][0],mutable_A[i][1]])
           
    for l in range(L):
        #Add complete dual problem
        
        #Objetive Constraint
        lhs = 0
        counterLEq = 0
        counterGEq = 0
        counterEq = 0
        for i in range(m):
            if sense_constrs[i]=="<":
                if i in mutable_b:
                    lhs = lhs + b[mutable_b.index(i)]*wLEq[l,counterLEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs + b_hat[i]*wLEq[l,counterLEq]    
                counterLEq+=1
            elif sense_constrs[i]==">":
                if i in mutable_b:
                    lhs = lhs + b[mutable_b.index(i)]*wGEq[l,counterGEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs + b_hat[i]*wGEq[l,counterGEq]    
                counterGEq+=1
            elif sense_constrs[i]=="=":
                if i in mutable_b:
                    lhs = lhs + b[mutable_b.index(i)]*wEq[l,counterEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs + b_hat[i]*wEq[l,counterEq]    
                counterEq+=1
                
        for j in range(n):
            if math.isinf(ub_vars[j]):
                val_ub = 0
            else:
                val_ub = ub_vars[j]
            if math.isinf(lb_vars[j]):
                val_lb = 0
            else:
                val_lb = lb_vars[j]
            
            if j in mutable_c:
                lhs = lhs + c[mutable_c.index(j)]*q[l,j]
            else:
                if c_hat[j]==0:
                    pass
                else:
                    lhs = lhs + c_hat[j]*q[l,j]
                
            lhs = lhs + val_ub*wu[l,j] + val_lb*wl[l,j] 
        
        approx_factor = 0.0   
        
        if list_bounds[l][0]>=0:
            lp.addConstr(lhs, sense = ">", rhs=(1-approx_factor)*list_bounds[l][0])
        else:
            lp.addConstr(lhs, sense = ">", rhs=(1+approx_factor)*list_bounds[l][0])
        
        #Gamma c + wA + wu +wl constraints
        for j in range(n):
            if j in mutable_c:
                lhs = gamma[l]*c[mutable_c.index(j)]
            else:
                if c_hat[j]==0:
                    pass
                else:
                    lhs = c_hat[j]*gamma[l]
                
            counterLEq = 0
            counterGEq = 0
            counterEq = 0
            for i in range(m):
                if sense_constrs[i]=="<":
                    if (i,j) in mutable_A:
                        lhs = lhs + A[mutable_A.index((i,j))]*wLEq[l,counterLEq]
                    else:
                        if A_hat[i,j]==0:
                            pass
                        else:
                            lhs = lhs + A_hat[i,j]*wLEq[l,counterLEq]    
                    counterLEq+=1
                elif sense_constrs[i]==">":
                    if (i,j) in mutable_A:
                        lhs = lhs + A[mutable_A.index((i,j))]*wGEq[l,counterGEq]
                    else:
                        if A_hat[i,j]==0:
                            pass
                        else:
                            lhs = lhs + A_hat[i,j]*wGEq[l,counterGEq]    
                    counterGEq+=1
                elif sense_constrs[i]=="=":
                    if (i,j) in mutable_A:
                        lhs = lhs + A[mutable_A.index((i,j))]*wEq[l,counterEq]
                    else:
                        if A_hat[i,j]==0:
                            pass
                        else:
                            lhs = lhs + A_hat[i,j]*wEq[l,counterEq]    
                    counterEq+=1
            
            if not math.isinf(ub_vars[j]):
                lhs = lhs + wu[l,j] 
                
            if not math.isinf(lb_vars[j]):
                lhs = lhs + wl[l,j] 
                
            
            if j==list_vars[l]:
                lp.addConstr(lhs, sense = "=", rhs=1)
            else:
                lp.addConstr(lhs, sense = "=", rhs=0)
        
        #by + qA constraints
        for i in range(m):
            if i in mutable_b:
                lhs = -b[mutable_b.index(i)]*gamma[l]
            else:
                if -b_hat[i]==0:
                    pass
                else:
                    lhs = -b_hat[i]*gamma[l]
                
            for j in range(n):
                if (i,j) in mutable_A:
                    lhs = lhs + A[mutable_A.index((i,j))]*q[l,j]
                else:
                    if A_hat[i,j]==0:
                        pass
                    else:
                        lhs = lhs + A_hat[i,j]*q[l,j]
                    
            if sense_constrs[i]=="<":
                lp.addConstr(lhs, sense = ">", rhs=0)
            elif sense_constrs[i]==">":
                lp.addConstr(lhs, sense = "<", rhs=0)
            elif sense_constrs[i]=="=":
                lp.addConstr(lhs, sense = "=", rhs=0)
                
                
        #u*gamma + q und l*gamma + q constraints
        for j in range(n):
            if not math.isinf(ub_vars[j]):
                lhs = -ub_vars[j]*gamma[l] + q[l,j]
                lp.addConstr(lhs, sense = ">", rhs=0)
                
            if not math.isinf(lb_vars[j]):
                lhs = -lb_vars[j]*gamma[l] + q[l,j]
                lp.addConstr(lhs, sense = "<", rhs=0)
                
    lp.write("StrongCEProblem.lp")
    lp.optimize()
    
    feasible = True
    no_solution_found = False
    hit_timelimit = False
    optimal_value = -1
    
    printStatus(lp.status)
    print("Number of solutions found:",lp.SolCount)
       
    if lp.status==GRB.INFEASIBLE:
        feasible = False
        no_solution_found = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.SolCount==0:
        no_solution_found = True
        if lp.status==GRB.TIME_LIMIT:
            hit_timelimit = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.status==GRB.TIME_LIMIT:
        hit_timelimit = True
        c_opt = np.array(list(lp.getAttr("X",c).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        A_opt = np.array(list(lp.getAttr("X",A).values()))
        x_ret = 0
        
        mu_c_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        mu_A_opt = np.array(list(lp.getAttr("X",mu_A).values()))
            
        nonZero_c = np.count_nonzero(mu_c_opt)
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_A = np.count_nonzero(mu_A_opt)
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = np.copy(A_hat)
        
        for i in range(n_c):
            c_ret[mutable_c[i]] = c_opt[i]
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            A_ret[mutable_A[i][0],mutable_A[i][1]] = A_opt[i]
        
        optimal_value = lp.ObjVal
    else:
        c_opt = np.array(list(lp.getAttr("X",c).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        A_opt = np.array(list(lp.getAttr("X",A).values()))
        x_ret = 0
        
        mu_c_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        mu_A_opt = np.array(list(lp.getAttr("X",mu_A).values()))
            
        nonZero_c = np.count_nonzero(mu_c_opt)
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_A = np.count_nonzero(mu_A_opt)
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        
        for i in range(n_c):
            c_ret[mutable_c[i]] = c_opt[i]
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            A_ret[mutable_A[i][0],mutable_A[i][1]] = A_opt[i]
        
        optimal_value = lp.ObjVal
        

    return feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_ret,A_ret,b_ret,x_ret,optimal_value   



def getWeakCounterfactual(n,m,c_hat,A_hat,b_hat,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b, valid_lb=0, x_feas=[]):
    approx_optimality = 1.0
    additive_optimality = 0.0
    lagrange_relaxation = False
    lagrange_multiplier = 0.5
    
    numLEq = 0
    numGEq = 0
    numEq = 0
    
    for sense in sense_constrs:
        if sense==">":
            numGEq+=1
        elif sense=="<":
            numLEq+=1
        elif sense=="=":
            numEq+=1
      
            
    n_c = len(mutable_c) 
    c_lb = np.zeros(n_c)
    c_ub = np.zeros(n_c)
    for j in range(n_c):
        if c_hat[mutable_c[j]]>=0:
            c_lb[j] = c_hat[mutable_c[j]]*(1-perc_c)
            c_ub[j] = c_hat[mutable_c[j]]*(1+perc_c)
        else:
            c_lb[j] = c_hat[mutable_c[j]]*(1+perc_c)
            c_ub[j] = c_hat[mutable_c[j]]*(1-perc_c)
    
    n_b = len(mutable_b)
    b_lb = np.zeros(n_b)
    b_ub = np.zeros(n_b)
    for i in range(n_b):
        if b_hat[mutable_b[i]]>=0:
            b_lb[i] = b_hat[mutable_b[i]]*(1-perc_b)
            b_ub[i] = b_hat[mutable_b[i]]*(1+perc_b)
        else:
            b_lb[i] = b_hat[mutable_b[i]]*(1+perc_b)
            b_ub[i] = b_hat[mutable_b[i]]*(1-perc_b)
    
    n_A = len(mutable_A)
    A_lb = np.zeros(n_A)
    A_ub = np.zeros(n_A)
    for i in range(n_A):
        if A_hat[mutable_A[i][0],mutable_A[i][1]]>=0:
            A_lb[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1-perc_A)
            A_ub[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1+perc_A)
        else:
            A_lb[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1+perc_A)
            A_ub[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1-perc_A)
    
    lp = gp.Model("WeakCE")
    lp.setParam("OutputFlag",1)
    lp.setParam("NonConvex",2)
    lp.setParam("FeasibilityTol",0.000000001)
    lp.setParam("OptimalityTol",0.000000001)
    #lp.setParam("Presolve",0)
    lp.setParam("TimeLimit",3600)
    
    lp.ModelSense=1
    
    #Add Variables
    x = lp.addVars(n,lb=lb_vars,ub=ub_vars, name="x")
    alpha = lp.addVars(n,ub=0.0, lb=-GRB.INFINITY, name="alpha")
    beta = lp.addVars(n,name="beta")
    yLEq = lp.addVars(numLEq,lb=-GRB.INFINITY,ub=0,name="y<")
    yGEq = lp.addVars(numGEq,name="y>")
    yEq = lp.addVars(numEq,lb=-GRB.INFINITY,name="y=")
    
    c = lp.addVars(n_c,lb=c_lb,ub=c_ub,name="c")
    A = lp.addVars(n_A, lb=A_lb, ub=A_ub, name="A")
    b = lp.addVars(n_b, lb=b_lb, ub=b_ub, name="b")
    
    mu_c = lp.addVars(n_c, obj = 1, name="mu_c")
    mu_b = lp.addVars(n_b, obj = 1, name="mu_b")
    mu_A = lp.addVars(n_A, obj = 1, name="mu_A")
    
    for j in range(n_c):
        lp.addConstr(mu_c[j]+c[j], sense = ">", rhs=c_hat[mutable_c[j]])
        lp.addConstr(mu_c[j]-c[j], sense = ">", rhs=-c_hat[mutable_c[j]])
        
    for i in range(n_b):
        lp.addConstr(mu_b[i]+b[i], sense = ">", rhs=b_hat[mutable_b[i]])
        lp.addConstr(mu_b[i]-b[i], sense = ">", rhs=-b_hat[mutable_b[i]])
        
    for i in range(n_A):
        lp.addConstr(mu_A[i]+A[i], sense = ">", rhs=A_hat[mutable_A[i][0],mutable_A[i][1]])
        lp.addConstr(mu_A[i]-A[i], sense = ">", rhs=-A_hat[mutable_A[i][0],mutable_A[i][1]])
        
    
    #add constraints c^t x <= c^t x_feas for solutions x_feas which are always feasible
    for i in range(len(x_feas)):
        lhs = 0
        rhs_val = 0
        for j in range(n):
            if j in mutable_c:
                lhs = lhs + c[mutable_c.index(j)]*x[j] - c[mutable_c.index(j)]*x_feas[i][j]
            else:
                lhs = lhs + c_hat[j]*x[j]
                rhs_val = rhs_val + c_hat[j]*x_feas[i][j]
                
        lp.addConstr(lhs,sense="<",rhs=rhs_val)
        
    #Add Ax<=b constraints
    for i in range(m):
        lhs = 0
        for j in range(n):
            if (i,j) in mutable_A:
                lhs = lhs + A[mutable_A.index((i,j))]*x[j]
            else:
                if A_hat[i,j]==0:
                    pass
                else:
                    lhs = lhs + A_hat[i,j]*x[j]
        
        if i in  mutable_b:
            lhs = lhs - b[mutable_b.index(i)]
            lp.addConstr(lhs, sense=sense_constrs[i], rhs=0)
        else:
            lp.addConstr(lhs, sense=sense_constrs[i], rhs=b_hat[i])
        
    #Add constraints for D(x)  
    counter = -1
    for j in list_vars:
        counter+=1
        if math.isinf(list_bounds[counter][1]):
            pass
        else:
            lp.addConstr(1*x[j], sense="<", rhs=list_bounds[counter][1])
            
        if math.isinf(list_bounds[counter][0]):
            pass
        else:
            lp.addConstr(1*x[j], sense=">", rhs=list_bounds[counter][0])
        
        
        
    #Add constraint: c^tx <= b^ty + u^t alpha + l^t beta
    if lagrange_relaxation:
        lhs = 0
        for i in range(n_c):
            lhs = lhs + mu_c[i]
        for i in range(n_b):
            lhs = lhs + mu_b[i]
        for i in range(n_A):
            lhs = lhs + mu_A[i]
            
        for j in range(n):
            if math.isinf(ub_vars[j]):
                val_ub = 0
            else:
                val_ub = ub_vars[j]
            if math.isinf(lb_vars[j]):
                val_lb = 0
            else:
                val_lb = lb_vars[j]
            
            if j in mutable_c:
                lhs = lhs + lagrange_multiplier*c[mutable_c.index(j)]*x[j]
            else:
                if c_hat[j]==0:
                    pass
                else:
                    lhs = lhs + lagrange_multiplier*c_hat[j]*x[j]
                
            lhs = lhs - lagrange_multiplier*val_ub*alpha[j] - lagrange_multiplier*val_lb*beta[j]
            
        counterLEq = 0
        counterGEq = 0
        counterEq = 0
        
        for i in range(m):
            if sense_constrs[i]=="<":
                if i in mutable_b:
                    lhs = lhs - lagrange_multiplier*b[mutable_b.index(i)]*yLEq[counterLEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs - lagrange_multiplier*b_hat[i]*yLEq[counterLEq] 
                counterLEq+=1
            elif sense_constrs[i]==">":
                if i in mutable_b:
                    lhs = lhs - lagrange_multiplier*b[mutable_b.index(i)]*yGEq[counterGEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs - lagrange_multiplier*b_hat[i]*yGEq[counterGEq]
                counterGEq+=1
            elif sense_constrs[i]=="=":
                if i in mutable_b:
                    lhs = lhs - lagrange_multiplier*b[mutable_b.index(i)]*yEq[counterEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs - lagrange_multiplier*b_hat[i]*yEq[counterEq]
                counterEq+=1

        lp.setObjective(lhs,GRB.MINIMIZE)      
    else:
        lhs = 0
        for j in range(n):
            if math.isinf(ub_vars[j]):
                val_ub = 0
            else:
                val_ub = ub_vars[j]
            if math.isinf(lb_vars[j]):
                val_lb = 0
            else:
                val_lb = lb_vars[j]
            
            if j in mutable_c:
                lhs = lhs + c[mutable_c.index(j)]*x[j]
            else:
                if c_hat[j] == 0:
                    pass
                else:
                    lhs = lhs + c_hat[j]*x[j]
                
            lhs = lhs - val_ub*alpha[j] - val_lb*beta[j]
            
        counterLEq = 0
        counterGEq = 0
        counterEq = 0
        
        
        for i in range(m):
            if sense_constrs[i]=="<":
                if i in mutable_b:
                    lhs = lhs - approx_optimality*b[mutable_b.index(i)]*yLEq[counterLEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs - approx_optimality*b_hat[i]*yLEq[counterLEq] 
                counterLEq+=1
            elif sense_constrs[i]==">":
                if i in mutable_b:
                    lhs = lhs - approx_optimality*b[mutable_b.index(i)]*yGEq[counterGEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs - approx_optimality*b_hat[i]*yGEq[counterGEq]
                counterGEq+=1
            elif sense_constrs[i]=="=":
                if i in mutable_b:
                    lhs = lhs - approx_optimality*b[mutable_b.index(i)]*yEq[counterEq]
                else:
                    if b_hat[i] == 0:
                        pass
                    else:
                        lhs = lhs - approx_optimality*b_hat[i]*yEq[counterEq]
                counterEq+=1
                
        lp.addConstr(lhs,sense="<",rhs=additive_optimality)
            
            
    #Add constraints: y^t A + alpha + beta = c
    for j in range(n):
        counterLEq = 0
        counterGEq = 0
        counterEq = 0
        if math.isinf(ub_vars[j]):
            val_alpha = 0
        else:
            val_alpha = 1
            
        if math.isinf(lb_vars[j]):
            val_beta = 0
        else:
            val_beta = 1
        
        lhs = val_alpha*alpha[j] + val_beta*beta[j]
        
        if j in mutable_c:
            lhs = lhs - 1*c[mutable_c.index(j)]
        
        for i in range(m):
            if sense_constrs[i]=="<":
                if (i,j) in mutable_A:
                    lhs = lhs + yLEq[counterLEq]*A[mutable_A.index((i,j))]
                else:
                    if A_hat[i,j]==0:
                        pass
                    else:
                        lhs = lhs + yLEq[counterLEq]*A_hat[i,j]
                counterLEq+=1
            elif sense_constrs[i]==">":
                if (i,j) in mutable_A:
                    lhs = lhs + yGEq[counterGEq]*A[mutable_A.index((i,j))]
                else:
                    if A_hat[i,j]==0:
                        pass
                    else:
                        lhs = lhs + yGEq[counterGEq]*A_hat[i,j]
                counterGEq+=1
            elif sense_constrs[i]=="=":
                if (i,j) in mutable_A:
                    lhs = lhs + yEq[counterEq]*A[mutable_A.index((i,j))]
                else:
                    if A_hat[i,j]==0:
                        pass
                    else:
                        lhs = lhs + yEq[counterEq]*A_hat[i,j]
                counterEq+=1
        
        if j in mutable_c:
            lp.addConstr(lhs,sense="=",rhs=0)
        else:
            lp.addConstr(lhs,sense="=",rhs=c_hat[j])
        
    lp.optimize()
    lp.write("WeakCEProblem.lp")
    feasible = True
    no_solution_found = False
    hit_timelimit = False
    optimal_value = -1
    
    printStatus(lp.status)
    print("Number of solutions found:",lp.SolCount)
       
    if lp.status==GRB.INFEASIBLE:
        feasible = False
        no_solution_found = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.SolCount==0:
        no_solution_found = True
        if lp.status==GRB.TIME_LIMIT:
            hit_timelimit = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.status==GRB.TIME_LIMIT:
        hit_timelimit = True
        c_opt = np.array(list(lp.getAttr("X",c).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        A_opt = np.array(list(lp.getAttr("X",A).values()))
        x_ret = np.array(list(lp.getAttr("X",x).values()))
        
        for i in range(len(c_opt)):
            if abs(c_opt[i])<0.0001:
                c_opt[i] = 0
                
        for i in range(len(b_opt)):
            if abs(b_opt[i])<0.0001:
                b_opt[i] = 0
                
        for i in range(len(A_opt)):
            if abs(A_opt[i])<0.0001:
                A_opt[i] = 0
        
        mu_c_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        mu_A_opt = np.array(list(lp.getAttr("X",mu_A).values()))
            
        nonZero_c = np.count_nonzero(mu_c_opt)
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_A = np.count_nonzero(mu_A_opt)
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        
        for i in range(n_c):
            c_ret[mutable_c[i]] = c_opt[i]
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            A_ret[mutable_A[i][0],mutable_A[i][1]] = A_opt[i]
        
        optimal_value = lp.ObjVal
        
    else:
        c_opt = np.array(list(lp.getAttr("X",c).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        A_opt = np.array(list(lp.getAttr("X",A).values()))
        x_ret = np.array(list(lp.getAttr("X",x).values()))
        
        print(np.array(list(lp.getAttr("X",alpha).values())))
        print(np.array(list(lp.getAttr("X",beta).values())))
        print(np.array(list(lp.getAttr("X",yGEq).values())))
        print(x_ret)
        print(c_opt)
        print(A_opt)
        
        for i in range(len(c_opt)):
            if abs(c_opt[i])<0.0001:
                c_opt[i] = 0
                
        for i in range(len(b_opt)):
            if abs(b_opt[i])<0.0001:
                b_opt[i] = 0
                
        for i in range(len(A_opt)):
            if abs(A_opt[i])<0.0001:
                A_opt[i] = 0
            
            
        mu_c_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        mu_A_opt = np.array(list(lp.getAttr("X",mu_A).values()))
            
        nonZero_c = np.count_nonzero(mu_c_opt)
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_A = np.count_nonzero(mu_A_opt)
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        
        for i in range(n_c):
            c_ret[mutable_c[i]] = c_opt[i]
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            A_ret[mutable_A[i][0],mutable_A[i][1]] = A_opt[i]
        
        optimal_value = lp.ObjVal
        

    return feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_ret,A_ret,b_ret,x_ret,optimal_value



def getRelativeCounterfactual(n,m,c_hat,A_hat,b_hat,opt_hat,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b,opt_approx=1, objective_type = "classic"): 
    n_c = len(mutable_c)
    c_lb = np.zeros(n_c)
    c_ub = np.zeros(n_c)
    for j in range(n_c):
        if c_hat[mutable_c[j]]>=0:
            c_lb[j] = c_hat[mutable_c[j]]*(1-perc_c)
            c_ub[j] = c_hat[mutable_c[j]]*(1+perc_c)
        else:
            c_lb[j] = c_hat[mutable_c[j]]*(1+perc_c)
            c_ub[j] = c_hat[mutable_c[j]]*(1-perc_c)
    
    n_b = len(mutable_b)
    b_lb = np.zeros(n_b)
    b_ub = np.zeros(n_b)
    for i in range(n_b):
        if b_hat[mutable_b[i]]>=0:
            b_lb[i] = b_hat[mutable_b[i]]*(1-perc_b)
            b_ub[i] = b_hat[mutable_b[i]]*(1+perc_b)
        else:
            b_lb[i] = b_hat[mutable_b[i]]*(1+perc_b)
            b_ub[i] = b_hat[mutable_b[i]]*(1-perc_b)
    
    n_A = len(mutable_A)
    A_lb = np.zeros(n_A)
    A_ub = np.zeros(n_A)
    for i in range(n_A):
        if A_hat[mutable_A[i][0],mutable_A[i][1]]>=0:
            A_lb[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1-perc_A)
            A_ub[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1+perc_A)
        else:
            A_lb[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1+perc_A)
            A_ub[i] = A_hat[mutable_A[i][0],mutable_A[i][1]]*(1-perc_A)
    
    lp = gp.Model("RelativeCE")
    lp.setParam("OutputFlag",0)
    lp.setParam("NonConvex",2)
    lp.setParam("FeasibilityTol",0.000000001)
    lp.setParam("OptimalityTol",0.000000001)
    #lp.setParam("Presolve",0)
    lp.setParam("TimeLimit",1800)
    
    lp.ModelSense=1
    
    #Add Variables
    x = lp.addVars(n,lb=lb_vars,ub=ub_vars, name="x")
    
    c = lp.addVars(n_c,lb=c_lb,ub=c_ub,name="c")
    A = lp.addVars(n_A, lb=A_lb, ub=A_ub, name="A")
    b = lp.addVars(n_b, lb=b_lb, ub=b_ub, name="b")
    
    mu_c = lp.addVars(n_c, name="mu_c")
    mu_b = lp.addVars(n_b, name="mu_b")
    mu_A = lp.addVars(n_A, name="mu_A")
    
    
    #set objective
    lhs = 0
    for j in range(n_c):
        if objective_type=="classic":
            lhs = lhs + 1*mu_c[j]
        elif objective_type=="x_times_delta":
            lhs = lhs + mu_c[j]*x[mutable_c[j]]
            
    for j in range(n_A):
        if objective_type=="classic":
            lhs = lhs + 1*mu_A[j]
        elif objective_type=="x_times_delta":
            lhs = lhs + mu_A[j]*x[mutable_A[j][1]]
            
    lp.setObjective(lhs,GRB.MINIMIZE)
    
    for j in range(n_c):
        lp.addConstr(1*mu_c[j]+1*c[j], sense = ">", rhs=c_hat[mutable_c[j]])
        lp.addConstr(1*mu_c[j]-1*c[j], sense = ">", rhs=-c_hat[mutable_c[j]])
        
    for i in range(n_b):
        lp.addConstr(1*mu_b[i]+1*b[i], sense = ">", rhs=b_hat[mutable_b[i]])
        lp.addConstr(1*mu_b[i]-1*b[i], sense = ">", rhs=-b_hat[mutable_b[i]])
        
    for i in range(n_A):
        lp.addConstr(1*mu_A[i]+1*A[i], sense = ">", rhs=A_hat[mutable_A[i][0],mutable_A[i][1]])
        lp.addConstr(1*mu_A[i]-1*A[i], sense = ">", rhs=-A_hat[mutable_A[i][0],mutable_A[i][1]])
    
    lhs = 0            
    for j in range(n):
        if j in mutable_c:
            lhs = lhs + c[mutable_c.index(j)]*x[j]
        else:
            if c_hat[j]==0:
                pass
            else:
                lhs = lhs + c_hat[j]*x[j]
            
    lp.addConstr(lhs,sense="<",rhs=opt_approx*opt_hat)
            
            
    for i in range(m):
        curr_row = A_hat.getrow(i)
        lhs = 0
        for ind, tup in enumerate(mutable_A):
            if tup[0] == i:
                lhs = lhs + A[ind]*x[tup[1]]
        for j in curr_row.indices:
            if (i,j) in mutable_A:
                pass
            else:
                lhs = lhs + A_hat[i,j]*x[j]
        
        if i in mutable_b:
            lhs = lhs - b[mutable_b.index(i)]
            lp.addConstr(lhs, sense=sense_constrs[i], rhs=0)
        else:
            lp.addConstr(lhs, sense=sense_constrs[i], rhs=b_hat[i])
            
        
    #Add constraints for D(x)     
    counter = -1
    for j in list_vars:
        counter+=1
        if math.isinf(list_bounds[counter][1]):
            pass
        else:
            lp.addConstr(1*x[j], sense="<", rhs=list_bounds[counter][1])
            
        if math.isinf(list_bounds[counter][0]):
            pass
        else:
            lp.addConstr(1*x[j], sense=">", rhs=list_bounds[counter][0])
        
        
    lp.optimize()
    lp.write("CEProblem.lp")
    feasible = True
    no_solution_found = False
    hit_timelimit = False
    optimal_value = -1
    
    printStatus(lp.status)
    print("Number of solutions found:",lp.SolCount)
       
    start = time.time()
    if lp.status==GRB.INFEASIBLE:
        feasible = False
        no_solution_found = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.SolCount==0:
        no_solution_found = True
        if lp.status==GRB.TIME_LIMIT:
            hit_timelimit = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.status==GRB.TIME_LIMIT:
        hit_timelimit = True
        c_opt = np.array(list(lp.getAttr("X",c).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        A_opt = np.array(list(lp.getAttr("X",A).values()))
        x_ret = np.array(list(lp.getAttr("X",x).values()))
        
        mu_c_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        mu_A_opt = np.array(list(lp.getAttr("X",mu_A).values()))
            
        nonZero_c = np.count_nonzero(mu_c_opt)
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_A = np.count_nonzero(mu_A_opt)
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        
        obj_value = 0
        for i in range(n_c):
            c_ret[mutable_c[i]] = c_opt[i]
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            A_ret[mutable_A[i][0],mutable_A[i][1]] = A_opt[i]
            
        optimal_value = lp.ObjVal
        
    else:
        c_opt = np.array(list(lp.getAttr("X",c).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        A_opt = np.array(list(lp.getAttr("X",A).values()))
        x_ret = np.array(list(lp.getAttr("X",x).values()))
        
        mu_c_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        mu_A_opt = np.array(list(lp.getAttr("X",mu_A).values()))
            
        nonZero_c = np.count_nonzero(mu_c_opt)
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_A = np.count_nonzero(mu_A_opt)
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        #A_ret = np.copy(A_hat)
        
        for i in range(n_c):
            c_ret[mutable_c[i]] = c_opt[i]
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            A_ret[mutable_A[i][0],mutable_A[i][1]] = A_opt[i]
        
        optimal_value = lp.ObjVal

        
    end = time.time()
    print("Saving solution of problem takes",end-start,"seconds.")
        
    return feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_ret,A_ret,b_ret,x_ret, optimal_value 
    
    
def getRelativeCounterfactualLinearization(n,m,c_hat,A_hat,b_hat,opt_hat,sense_constrs,lb_vars,ub_vars,list_vars,list_bounds,mutable_c,perc_c,mutable_A, perc_A, mutable_b, perc_b, opt_approx=1, objective_type = "x_times_delta"): 
            
    n_c = len(mutable_c) 
    n_b = len(mutable_b)
    n_A = len(mutable_A)
    
    lp = gp.Model("RelativeCE")
    lp.setParam("OutputFlag",0)
    lp.setParam("FeasibilityTol",0.000001)
    lp.setParam("OptimalityTol",0.000001)
    lp.setParam("TimeLimit",1800)
    
    lp.ModelSense=1
    
    #Add Variables
    x = lp.addVars(n,lb=lb_vars,ub=ub_vars, name="x")
    
    w = lp.addVars(n_c,lb=-GRB.INFINITY,name="w")
    U = lp.addVars(n_A,lb=-GRB.INFINITY,name="U")
    b = lp.addVars(n_b,lb=-GRB.INFINITY,name="b")
    
    mu_c = lp.addVars(n_c, obj = 1, name="mu_c")
    mu_b = lp.addVars(n_b, obj = 1, name="mu_b")
    mu_A = lp.addVars(n_A, obj = 1, name="mu_A")

    

    for j in range(n_c):
        lp.addConstr(mu_c[j]+w[j]-c_hat[mutable_c[j]]*x[mutable_c[j]], sense = ">", rhs=0)
        lp.addConstr(mu_c[j]-w[j]+c_hat[mutable_c[j]]*x[mutable_c[j]], sense = ">", rhs=0)

        
    for i in range(n_b):
        lp.addConstr(mu_b[i]+b[i], sense = ">", rhs=b_hat[mutable_b[i]])
        lp.addConstr(mu_b[i]-b[i], sense = ">", rhs=-b_hat[mutable_b[i]])
        
    for i in range(n_A):
        lp.addConstr(mu_A[i]+U[i]-A_hat[mutable_A[i][0],mutable_A[i][1]]*x[mutable_A[i][1]], sense = ">", rhs=0)
        lp.addConstr(mu_A[i]-U[i]+A_hat[mutable_A[i][0],mutable_A[i][1]]*x[mutable_A[i][1]], sense = ">", rhs=0)

    
    #bounds w and U
    for j in range(n_c):
        if c_hat[mutable_c[j]]>=0:
            lp.addConstr(1*w[j] - (1-perc_c)*c_hat[mutable_c[j]]*x[mutable_c[j]], sense = ">", rhs=0)
            lp.addConstr(1*w[j] - (1+perc_c)*c_hat[mutable_c[j]]*x[mutable_c[j]], sense = "<", rhs=0)

            
        elif c_hat[mutable_c[j]]<0:
            lp.addConstr(1*w[j] - (1-perc_c)*c_hat[mutable_c[j]]*x[mutable_c[j]], sense = "<", rhs=0)
            lp.addConstr(1*w[j] - (1+perc_c)*c_hat[mutable_c[j]]*x[mutable_c[j]], sense = ">", rhs=0)

        
    for i in range(n_A):
        if A_hat[mutable_A[i][0],mutable_A[i][1]]>=0:
            lp.addConstr(1*U[i] - (1-perc_A)*A_hat[mutable_A[i][0],mutable_A[i][1]]*x[mutable_A[i][1]], sense = ">", rhs=0)
            lp.addConstr(1*U[i] - (1+perc_A)*A_hat[mutable_A[i][0],mutable_A[i][1]]*x[mutable_A[i][1]], sense = "<", rhs=0)

        
        elif A_hat[mutable_A[i][0],mutable_A[i][1]]<0:
            lp.addConstr(1*U[i] - (1-perc_A)*A_hat[mutable_A[i][0],mutable_A[i][1]]*x[mutable_A[i][1]], sense = "<", rhs=0)
            lp.addConstr(1*U[i] - (1+perc_A)*A_hat[mutable_A[i][0],mutable_A[i][1]]*x[mutable_A[i][1]], sense = ">", rhs=0)
      
         
    lhs = 0            
    for j in range(n):
        if j in mutable_c:
            lhs = lhs + 1*w[mutable_c.index(j)]
        else:
            if c_hat[j] == 0:
                pass
            else:
                lhs = lhs + c_hat[j]*x[j]
            
    lp.addConstr(lhs,sense="<",rhs=opt_approx*opt_hat)


    
    for i in range(m):
        curr_row = A_hat.getrow(i)
        lhs = 0
        for ind, tup in enumerate(mutable_A):
            if tup[0] == i:
                lhs = lhs + U[ind]
        for j in curr_row.indices:
            if (i,j) in mutable_A:
                pass
            else:
                lhs = lhs + A_hat[i,j]*x[j]
        
        if i in mutable_b:
            lhs = lhs - b[mutable_b.index(i)]
            lp.addConstr(lhs, sense=sense_constrs[i], rhs=0)
        else:
            lp.addConstr(lhs, sense=sense_constrs[i], rhs=b_hat[i])
    
    #Add constraints for D(x)     
    counter = -1
    for j in list_vars:
        counter+=1
        if math.isinf(list_bounds[counter][1]):
            pass
        else:
            lp.addConstr(1*x[j], sense="<", rhs=list_bounds[counter][1])
            
        if math.isinf(list_bounds[counter][0]):
            pass
        else:
            lp.addConstr(1*x[j], sense=">", rhs=list_bounds[counter][0])
        
    
    lp.optimize()

    #lp.write("CEProblem_Linearization.lp")
    feasible = True
    no_solution_found = False
    hit_timelimit = False
    optimal_value = -1
    
    printStatus(lp.status)
    print("Number of solutions found:",lp.SolCount)
    
    if lp.status==GRB.INFEASIBLE:
        feasible = False
        no_solution_found = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.SolCount==0:
        no_solution_found = True
        if lp.status==GRB.TIME_LIMIT:
            hit_timelimit = True
        c_ret = 0
        A_ret = 0
        b_ret = 0
        x_ret = 0
        
        nonZero_c = 0
        nonZero_b = 0
        nonZero_A = 0
    elif lp.status==GRB.TIME_LIMIT:
        hit_timelimit = True
        w_opt = np.array(list(lp.getAttr("X",w).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        U_opt = np.array(list(lp.getAttr("X",U).values()))
        x_ret = np.array(list(lp.getAttr("X",x).values()))
        
        #mu_w_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        #mu_U_opt = np.array(list(lp.getAttr("X",mu_A).values()))
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_c = 0
        nonZero_A = 0
        
        optimal_value = 0
        
        for i in range(n_c):
            if x_ret[mutable_c[i]]==0:
                pass
            else:
                c_ret[mutable_c[i]] = w_opt[i]/x_ret[mutable_c[i]]
                optimal_value = optimal_value + abs(c_ret[mutable_c[i]]-c_hat[mutable_c[i]])
                if abs(w_opt[i]/x_ret[mutable_c[i]]-c_hat[mutable_c[i]])>0.00001:
                    nonZero_c+=1
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            if x_ret[mutable_A[i][1]]==0:
                pass
            else:
                A_ret[mutable_A[i][0],mutable_A[i][1]] = U_opt[i]/x_ret[mutable_A[i][1]]
                optimal_value = optimal_value + abs(A_ret[mutable_A[i][0],mutable_A[i][1]]-A_hat[mutable_A[i][0],mutable_A[i][1]])
                if abs(U_opt[i]/x_ret[mutable_A[i][1]]-A_hat[mutable_A[i][0],mutable_A[i][1]])>0.00001:
                    nonZero_A+=1
        
    else:
        w_opt = np.array(list(lp.getAttr("X",w).values()))
        b_opt = np.array(list(lp.getAttr("X",b).values()))
        U_opt = np.array(list(lp.getAttr("X",U).values()))
        x_ret = np.array(list(lp.getAttr("X",x).values()))
        
        #mu_w_opt = np.array(list(lp.getAttr("X",mu_c).values()))
        mu_b_opt = np.array(list(lp.getAttr("X",mu_b).values()))
        #mu_U_opt = np.array(list(lp.getAttr("X",mu_A).values()))
        
        c_ret = np.copy(c_hat)
        b_ret = np.copy(b_hat)
        A_ret = sp.csr_matrix(A_hat).copy()
        
        nonZero_b = np.count_nonzero(mu_b_opt)
        nonZero_c = 0
        nonZero_A = 0
        
        optimal_value = 0
        
        for i in range(n_c):
            if x_ret[mutable_c[i]]==0:
                pass
            else:
                c_ret[mutable_c[i]] = w_opt[i]/x_ret[mutable_c[i]]
                optimal_value = optimal_value + abs(c_ret[mutable_c[i]]-c_hat[mutable_c[i]])
                if abs(w_opt[i]/x_ret[mutable_c[i]]-c_hat[mutable_c[i]])>0.00001:
                    nonZero_c+=1
            
        for i in range(n_b):
            b_ret[mutable_b[i]] = b_opt[i]
            
        for i in range(n_A):
            if x_ret[mutable_A[i][1]]==0:
                pass
            else:
                A_ret[mutable_A[i][0],mutable_A[i][1]] = U_opt[i]/x_ret[mutable_A[i][1]]
                optimal_value = optimal_value + abs(A_ret[mutable_A[i][0],mutable_A[i][1]]-A_hat[mutable_A[i][0],mutable_A[i][1]])
                if abs(U_opt[i]/x_ret[mutable_A[i][1]]-A_hat[mutable_A[i][0],mutable_A[i][1]])>0.00001:
                    nonZero_A+=1

    return feasible,no_solution_found,hit_timelimit,nonZero_c,nonZero_b,nonZero_A,c_ret,A_ret,b_ret,x_ret, optimal_value 
    
    
def checkRelativeCECorrect(n,m,c,A,b,sense_constrs,lb_vars,ub_vars,model_sense,list_vars,list_bounds,objBound):
    lp = gp.Model("CheckRelativeCE")
    lp.setParam("OutputFlag",0)
    lp.setParam("TimeLimit",7200)
    
    lp.ModelSense = model_sense
    x = lp.addVars(n,obj=c,vtype=GRB.CONTINUOUS,lb=lb_vars,ub=ub_vars, name="x")
    
    
    for i in range(A.shape[0]):
        lhs = 0
        for j in range(n):
            lhs = lhs + A[i,j]*x[j]
        lp.addConstr(lhs, sense=sense_constrs[i], rhs=b[i])
        
    #Add constraints for D(x)     
    counter = -1
    for j in list_vars:
        counter+=1
        if math.isinf(list_bounds[counter][1]):
            pass
        else:
            lp.addConstr(1*x[j], sense="<", rhs=list_bounds[counter][1])
            
        if math.isinf(list_bounds[counter][0]):
            pass
        else:
            lp.addConstr(1*x[j], sense=">", rhs=list_bounds[counter][0])
        
    lp.optimize()
    #lp.write("CheckCEProblem.lp")
    
    correct = False
    if lp.status==GRB.INFEASIBLE:
        print("CE not correct, infeasible")
        correct = False
    elif lp.status == GRB.OPTIMAL and lp.ObjVal <=objBound+0.0001:
        print("CE Correct")
        print("Optimal Value:",lp.ObjVal)
        print("To achieve bound:",objBound)
        correct = True
    else:
        print("CE not correct, else")
        print("Status:",lp.status)
        print("Optimal Value:",lp.ObjVal)
        print("To achieve bound:",objBound)
        correct = False
    
    return correct
    
