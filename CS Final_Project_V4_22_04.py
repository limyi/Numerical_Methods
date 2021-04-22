# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:44:57 2021

@author: Lim Yi, Ash, Glenn, Mani Singaopore University of Technology and Design
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.linalg 
e = np.exp(1)


'-----------------------------Import Config-----------------------------------'
data = pd.read_csv("CS_Project_Param_File.csv") 

Difference_method = np.array(data['Difference_Method'][0:1])
Difference_Method_Step_size = np.array(data['Difference_Method_Step_size'][0:1])
A = np.array(data['Amplitude'][0:1])  
wd =np.array(data['Damp_W'][0:1])    
wn = np.array(data['Natural_w'][0:1]) 
L = np.array(data['Damping_Ratio'][0:1]) 
m = np.array(data['Mass_Nb'][0:1]) 
k = np.array(data['Stiffness_Nb'][0:1]) 
Number_Of_Nodes =  int(np.array(data['Number_Of_Nodes'][0:1]))
Quality_Factor = np.array(data['Quality_Factor'][0:1]) 
Matrix_Solver_Input = np.array(data['Matrix_Solver'][0:1])
Inverse_Method = np.array(data['Inverse_Method'][0:1]) 
Root_Finding_Approach   = np.array(data['Root_Finding_Approach'][0:1]) 
root_finding = data['Root_Finding'].tolist()
print('Enter the time of Interest: ')
t = int(input())
print('Enter the location point of Interest: ')
deltaX = int(input())


if len(root_finding):
        lower = root_finding[0]
        upper = root_finding[1]
        x0 = root_finding[2]
        x1 = root_finding[3]
        x2 = root_finding[4]
        error = root_finding[5]
        N = root_finding[6]
        x_i = root_finding[7]
        x_p = root_finding[8]
        initial_guess = root_finding[9]
else:
    print("Invalid Root Finding on csv file")



'----------------------------Compute Config-----------------------------------'
L = 1/(2*Quality_Factor)
b = L * 2 * np.sqrt(m*k)


'''------------------------Find Root of Vel Eqn-----------------------------'''
'-----------------------------Root Solver 1-----------------------------------'
def bracket(func, lower, upper,error,args = '0'):
            err = 100
            xr_old = lower + upper
            i = 0
            err_list = []
            x_list = []
            
            while err>error:
                xr = (lower + upper) / 2
                l = func(lower,A,wd,wn,L)
                c = func(xr,A,wd,wn,L)
                if l*c<0:
                    upper = xr
                else:
                    lower = xr
                err = 100 * abs((xr-xr_old)/xr)
                err_list.append(err)
                x_list.append(xr)
                # print (i,xr,err)
                xr_old = xr
                i = i + 1
            
            return xr,x_list,err_list,i
        
'-----------------------------Root Solver 2-----------------------------------'
def Muller(func,x0,x1,x2,n,e):
    i = 1
    error = 100
    e_r = []
    x_r = []
    while i<n and error>e:
        h0 = x1-x0
        h1 = x2-x1
        del_0 = (func(x1,A,wd,wn,L) - func(x0,A,wd,wn,L))/((x1-x0))
        del_1 = (func(x2,A,wd,wn,L) - func(x1,A,wd,wn,L))/((x2-x1))
        a = (del_1 - del_0)/ (h1+h0)
        b = (a*h1) + (del_1)
        c = func(x2,A,wd,wn,L)
        rad = np.sqrt((b**2)- (4*a*c))
        if np.abs(b+rad) > np.abs(b-rad):
            den = b+rad
        else:
            den = b-rad
        x3 = x2 + ((-2*c)/(den))
        x_r.append(x3)
        error = 100 * np.abs((x3-x2)/x3)
        e_r.append(error)
        # print("Number of Iteration",i,"Root_Value",x3,"Error is",error)
        x0 = x1
        x1 = x2
        x2 = x3
        i = i+1
    return x3,x_r,e_r,i

'-----------------------------Root Solver 3-----------------------------------'
def isNaN(num):
    #checks if num is a NaN
    return num != num

def secant(func,x_i,x_p,n_max):    
    i = 0
    ea = 10
    x_r = []
    e_r = []
    while i<n_max: #and ea>error:
        if isNaN(x_i)==True: #checks for NaNs, which occur after the system has converged on a root
            #print ("Root found at", x_p, "after", i-1, "itterations.")
            x_i=x_p
            break #stops the loop if a root is found
        x_updated = x_i-((func(x_i,A,wd,wn,L)*(x_p-x_i))/(func(x_p,A,wd,wn,L)-func(x_i,A,wd,wn,L)))
        x_r.append(x_updated)
        ea = 100 * np.abs((x_updated - x_p)/(x_updated))
        e_r.append(ea)
        # print (i,x_p,x_i,x_updated,ea)
        x_p = x_i
        x_i = x_updated
        i += 1
    return x_updated,x_r,e_r,i

'-----------------------------Root Solver 4-----------------------------------'
def function_for_Raphson(t):
    f_x = (A*wd*e**(-L*wn*t)*np.cos(wd*t)-A*L*wn*e**(-L*wn*t)*np.sin(wd*t))
    ff_x = (-A*wd**2*e**(-L*wn*t)*np.sin(wd*t)+A*L**2*wn**2*e**(-L*wn*t)*np.sin(wd*t)-2*A*L*wn*wd*e**(-L*wn*t)*np.cos(wd*t))
    fff_x = 3*A*L*wn*wd**2*e**(-L*wn*t)*np.sin(wd*t)-A*L**3*wn**3*e**(-L*wn*t)*np.sin(wd*t)-A*wd**3*e**(-L*wn*t)*np.cos(wd*t)+3*A*L**2*wn**2*wd*e**(-L*wn*t)*np.cos(wd*t)
    Zeroth_M = t - ((f_x*ff_x)/(((ff_x)**2) - (f_x*fff_x)))
    return Zeroth_M
    

def raphson(func,initial_guess,error,max_iter):
    
    i = 0
    ea = 10
    xr = []
    err = []
    
    while i<max_iter and ea>error:
        x_new = func(initial_guess)
        xr.append(x_new)
        ea = 100*np.abs((x_new - initial_guess)/x_new)
        err.append(ea)
        # print(i,initial_guess,x_new,ea)
        initial_guess = x_new
        i += 1
    return (x_new,xr,err,i)


'----------------------------Differentiation----------------------------------'
def Damp_SHM_displacement(t,A,wd,wn,L):
    return A*e**(-L*wn*t)*np.sin(wd*t)

def Damp_SHM_velocity(t,A,wd,wn,L):
    return A*(-L*wn)*e**(-L*wn*t)*np.sin(wd*t)+A*e**(-L*wn*t)*wd*np.cos(wd*t) 

def CD(func, t,A,wd,wn,L,h):  #Central Difference
    dydt=(func(t+h,A,wd,wn,L)-func(t-h,A,wd,wn,L))/(2*h)
    return dydt

def FD(func, t,A,wd,wn,L, h):  #Forward Difference
    dydt=(func(t+h,A,wd,wn,L)-func(t,A,wd,wn,L))/(h)
    return dydt

def BD(func, t,A,wd,wn,L, h):  #Backward Difference
    dydt=(func(t,A,wd,wn,L)-func(t-h,A,wd,wn,L))/(h)
    return dydt

def CFD(func, t,A,wd,wn,L, h):  #Central Finite Divided
    dydt=( -func(t+2*h,A,wd,wn,L) + 8* func(t+h,A,wd,wn,L) - 8* func(t-h,A,wd,wn,L) + func(t-2*h,A,wd,wn,L)) / (12*h)
    return dydt

def FFD(func, t,A,wd,wn,L, h):  #Forward Finite Divided
    dydt=( -func(t+2*h,A,wd,wn,L) + 4*func(t+h,A,wd,wn,L) - 3*func(t,A,wd,wn,L) )/(2*h)
    return dydt

def BFD(func, t,A,wd,wn,L, h):  #Backward Finite Divided
    dydt=( 3* func(t,A,wd,wn,L) - 4*func(t-h,A,wd,wn,L) + func(t-2*h,A,wd,wn,L) )/(2*h)
    return dydt

def F1(Difference_Method_Step_size,t,A,wd,wn,L,m):
    print("This is running")
    if Difference_method == "Forward_Difference":
        print("Forward_Difference")
        return FD(Damp_SHM_velocity,t,A,wd,wn,L,Difference_Method_Step_size)*m
    elif Difference_method == "Backward_Difference":
        print("Backward_Difference")
        return BD(Damp_SHM_velocity,t,A,wd,wn,L,Difference_Method_Step_size)*m
    elif Difference_method == "Central_Difference":
        print("Central_Difference")
        return CD(Damp_SHM_velocity,t,A,wd,wn,L,Difference_Method_Step_size)*m
    elif Difference_method == "Forward_Finite_Difference":
        print("Forward_Finite_Difference")
        return FFD(Damp_SHM_velocity,t,A,wd,wn,L,Difference_Method_Step_size)*m
    elif Difference_method == "Backward_Finite_Difference":
        print("Backward_Finite_Difference")
        return BFD(Damp_SHM_velocity,t,A,wd,wn,L,Difference_Method_Step_size)*m
    elif Difference_method == "Central_Finite_Difference":
        print("Central_Finite_Difference")
        return CFD(Damp_SHM_velocity,t,A,wd,wn,L,Difference_Method_Step_size)*m
    else:
        print("Please check possible inputs")
        return("NAN")
'------------------------------Main Action------------------------------------'
x = np.linspace(0,10*(8)*np.pi,100)
z = Damp_SHM_displacement(x,A,wd,wn,L)
y = Damp_SHM_velocity(x,A,wd,wn,L)
plt.figure()
plt.plot(x,y,"r",x,z, "--")
F_1 = F1(Difference_Method_Step_size,t,A,wd,wn,L,m)
print (F_1)
t_1 = 1
'''-----------------------Finite Element Approach---------------------------'''
'----------------------------Matrix Formation---------------------------------'
def Matrix_F_and_R(Number_of_node,F1,Friction,t,b,k):
    scalar = 1 #b/t*k
    Mat_R = np.zeros((Number_of_node,Number_of_node))
    Mat_F = np.zeros((Number_of_node,1))
    Mat_F[0] = F1

    for i in range(1,Number_of_node):
        Mat_F[i] = Friction
    
    for i in range(Number_of_node-1):
        for j in range(Number_of_node-1):
            if i == j:
                Mat_R[i][j] = 2
                Mat_R[i+1][j] = -1
                Mat_R[i][j+1] = -1
    Mat_R[0][0] = 1
    Mat_R[0][1] = -1
    Mat_R[Number_of_node-1][Number_of_node-1] = 1-1-1*10**-12
    Mat_R[Number_of_node-1][Number_of_node-2] = -1
    Mat_R = scalar * Mat_R
    return Mat_R , Mat_F
'--------------------------Checking Conditions--------------------------------'
def Inverse(Inverse_Method,Matrix_R):
    print("This is running")
    if Inverse_Method == "Numpy_INV":
        print("Numpy_INV")
        inva = np.linalg.inv(Matrix_R)
        Identity = np.linalg.inv(Matrix_R).dot(Matrix_R)
        return inva , Identity
    elif Inverse_Method == "Inverse_LU_Method":
        print("Inverse_LU_Method")
        inva = Inverse_LU_Method(Matrix_R,Number_Of_Nodes)
        Identity = np.dot(inva,Matrix_R)
        return inva , Identity
    elif Inverse_Method == "Inverse_Gauss_Jordan":
        print("Central_Difference")
        inva =Inverse_Gauss_Jordan(Matrix_R,Number_Of_Nodes)
        Identity = np.dot(inva,Matrix_R)
        return inva , Identity
 #'--------------------------------------------------------------------------------help-' 
        return CFD(Damp_SHM_velocity,Difference_Method_Step_size,t,A,wd,wn,L)
    else:
        print("Please check possible inputs")
        return("NAN")

def Inverse_LU_Method(Matrix_R,Number_Of_Nodes):
        lower,upper = LUMETHOD(Matrix_R,Number_Of_Nodes)
        identity = np.identity(Number_Of_Nodes)
        LU_List = []
        for i in range(Number_Of_Nodes):
            column =identity[:,i]
            column = column.reshape((Number_Of_Nodes,1))
            Matrix_D = forward_sub(lower,column)
            Matrix_X = backward_sub(upper,Matrix_D)
            temp_list = []
            for i in range(len(Matrix_X)):
                temp_list.append(float(Matrix_X[i]))
            LU_List.append(temp_list)
        return LU_List

def LUMETHOD(Matrix, size): #LU Decomposition for Matrix of size = size
 
    lower = [[0 for x in range(size)]for y in range(size)]  #Create empty matrix of size by size
    upper = [[0 for x in range(size)]for y in range(size)]  #Create empty matrix of size by size
    

    # Decomposing Matrix into Upper and Lower triangular matrix
    for x in range(size):
 
        # Upper Triangular
        for y in range(x, size):
                
            sum = 0
            for z in range(x):
#                print(x,y,z)
                sum += (lower[x][z] * upper[z][y])
#                print(lower[x][z],upper[z][y],sum)
            # Evaluating U(x, y)
            upper[x][y] = Matrix[x][y] - sum
 
        # Lower Triangular
        
        for y in range(x, size):
            if (x == y):
                lower[x][x] = 1  # The Diagonal of Lower Triangular is 1
            else:
 
                # Summatxon of L(y, z) * U(z, x)
                sum = 0
                for z in range(x):
                    sum += (lower[y][z] * upper[z][x])
 
                # Evaluating L(y, x)

                lower[y][x] = (Matrix[y][x] - sum) / upper[x][x]
    return lower,upper

def forward_sub(L,b):
    n=len(L)
    d=[0]*n
    for i in range(n):
        j=0
        total=0
        for c in range(n-1):            
            if j==i:
                j=j+1
            total=total+L[i][j]*d[j]
            j=j+1
    
            d[i]=(b[i]-total)/L[i][i]         
    return d

def backward_sub(U,d):
    n=len(U)
    x=[0]*n
    for i in range (n):
        # print(i)
        j=n-1
        total=0
        for c in range (n-1):
            if j==(n-1-i):
                j=j-1
            total=total+U[n-i-1][j]*x[j]
            j=j-1

        x[n-i-1]=(d[n-i-1]-total)/U[n-i-1][n-i-1]
    return x

def Inverse_Gauss_Jordan(a,n):
    a = a.tolist()
    for i in range(n):
        for j in range(n):
            a[i].append(0.0)

    for i in range(n):        
        for j in range(n):
            if i == j:
                a[i][j+n] = 1

# Applying Guass Jordan Elimination
    for i in range(n):
        if a[i][i] == 0.0:
            print('Divide by zero detected!')
            
        for j in range(n):
            if i != j:
                ratio = a[j][i]/a[i][i]
    
                for k in range(2*n):
                    a[j][k] = a[j][k] - ratio * a[i][k]

# Row operation to make principal diagonal element to 1
    for i in range(n):
        divisor = a[i][i]
        for j in range(2*n):
            a[i][j] = a[i][j]/divisor

    for i in range(Number_Of_Nodes):

        for j in range(Number_Of_Nodes):
            del(a[i][0])
    return a

'-----------------------------Matrix Solving----------------------------------'
def Matrix_Solver(Matrix_Solver,Matrix_R,Matrix_F,Number_Of_Nodes):
    Number_Of_Nodes = Number_Of_Nodes
    Matrix_F = Matrix_F
    Matrix_R = Matrix_R
    Matrix_Solver = Matrix_Solver
    print("This is running",Matrix_Solver)
    X_Axis = []
    for i in range(Number_Of_Nodes):
        X_Axis.append(i)
    if Matrix_Solver == 'LU_Method':
        print("LU_Method")
        lower,upper = LUMETHOD(Matrix_R,Number_Of_Nodes)
        Matrix_D = forward_sub(lower,Matrix_F)
        Matrix_X = backward_sub(upper,Matrix_D)
#        plt.plot(X_Axis,Matrix_X)
        return Matrix_X
    elif Matrix_Solver == "Gauss_Jordan_Method":
        print("Gauss_Jordan_Method")
        Matrix_X = Gauss_Jordan(Matrix_R,Matrix_F,Number_Of_Nodes)
        # plt.plot(X_Axis,Matrix_X)
        return Matrix_X
    elif Matrix_Solver == "Numpy_Inverse_Solve":
        print("Numpy_Inverse_Solve")
        Matrix_X = np.linalg.inv(Matrix_R).dot(Matrix_F)
        # plt.plot(X_Axis,Matrix_X)
        return Matrix_X
    elif Matrix_Solver == "Numpy_Solve":
        print("Numpy_Solve")
        Matrix_X = np.linalg.solve(Matrix_R,Matrix_F)
        # plt.plot(X_Axis,Matrix_X)
        return Matrix_X
    elif Matrix_Solver == "scipy_LU_Method":
        print("scipy_LU_Method")
        P, L, U = scipy.linalg.lu(Matrix_R)
        v2 = np.linalg.solve(L,Matrix_F)
        Matrix_X = np.linalg.solve(U,v2)
        # plt.plot(X_Axis,Matrix_X)
        return Matrix_X
    elif Matrix_Solver == "Gauss_Elimination":
        print("Gauss_Elimination")
        Matrix_X = gauss_elimination(Matrix_R,Matrix_F,Number_Of_Nodes)
        # plt.plot(X_Axis,Matrix_X)
        return Matrix_X
        print("Please check possible inputs in Matrix Solver")
        return("NAN")

def Root_finding_approach(Root_Finding_Approach,lower, upper,error,A,wd,wn,L):
    print(Root_Finding_Approach)
#    Root_Finding_Approach = Root_Finding_Approach.lower()
    if Root_Finding_Approach == 'Bi-Section':
        print("Executing Bi-section")
        root, x_list,err_list,i = bracket(Damp_SHM_velocity, lower, upper,error,args = '0')
        print("Root = " + str(root))
        print("Converges after " + str(i-1),"iterations")
        return root, x_list,err_list,i
    elif Root_Finding_Approach == 'Muller':
        print("Executing Muller")
        root,x_list_M,err_list_M,i = Muller(Damp_SHM_velocity,x0,x1,x2,N,error)
        print("Root = " + str(root))
        print("Converges after " + str(i-1),"iterations")
        return root,x_list_M,err_list_M,i
    elif Root_Finding_Approach == 'Secant':
        print("Executing Secant")
        root,x_list_S,err_list_S,i = secant(Damp_SHM_velocity,x_i,x_p,N)
        print("Root = " + str(root))
        print("Converges after " + str(i-1),"iterations")
        return root,x_list_S,err_list_S,i
    elif Root_Finding_Approach == 'Newton-Raphson':
        print("Executing Modified Newton-Raphson")
        root,x_list_N,err_list_N,i = raphson(function_for_Raphson,initial_guess,error,N)
        print("Root = " + str(root))
        print("Converges after " + str(i-1),"iterations")
        return root,x_list_N,err_list_N,i
    else:
        print ("Please check Input") 
    
def Gauss_Jordan(Matrix_R,Matrix_F,n):
    Matrix_R = Matrix_R.tolist() 
    Matrix_F = Matrix_F.tolist() 
    x = np.zeros(n)
    for i in range(len(Matrix_F)):

        Matrix_R[i] = Matrix_R[i] + Matrix_F[i]
 
    for i in range(n):
        if Matrix_R[i][i] == 0.0:
            print('Singularity detected!')
        
        for j in range(n):
            if i != j:
                ratio = Matrix_R[j][i]/Matrix_R[i][i]
    
                for k in range(n+1):

                    Matrix_R[j][k] = Matrix_R[j][k] - ratio * Matrix_R[i][k]
    for i in range(n):
        x[i] = Matrix_R[i][n]/Matrix_R[i][i]
    return x


def gauss_elimination(x,y,n):
    x=np.copy(x)
    y=np.copy(y)
    n=len(y)
    c=np.zeros(n,float)
    for k in range (n):
        for i in range (k+1,n):
            m=x[i,k]/x[k,k]
            for j in range (k,n):
                x[i,j]=x[i,j]-m*x[k,j]
            y[i]= y[i]-m*y[k]

            
    c[n-1]=y[n-1]/x[n-1,n-1]
    for i in range (n-2,-1,-1):
        total=0
        for j in range (i+1,n):    
            total= total+ x[i,j]*c[j]
        c[i]=(y[i]-total)/x[i,i]
    return (c)
print('x',x)
'------------------------------Main Action------------------------------------'
Friction = 0.0#/(Number_Of_Nodes-1)
Matrix_R, Matrix_F = Matrix_F_and_R(Number_Of_Nodes,F_1,Friction,t,b,k)
Inverse_Matrix, Identity_Matrix= Inverse(Inverse_Method,Matrix_R)
#print(Matrix_R,"Matrix_R","\n")
#print(Matrix_F,"Matrix_F")


'------------------------------Main Action------------------------------------'
print('Solved')
x = Matrix_Solver(Matrix_Solver_Input,Matrix_R,Matrix_F,Number_Of_Nodes)
Inverse_Matrix, Identity_Matrix= Inverse(Inverse_Method,Matrix_R)
#print(Inverse_Matrix, Identity_Matrix)
Root_finding_approach(Root_Finding_Approach,lower, upper,error,A,wd,wn,L)

'''
def CD(func, t,A,wd,wn,L,h):  #Central Difference
    dydt=(func(t+h,A,wd,wn,L)-func(t-h,A,wd,wn,L))/(2*h)
    return dydt
'''

hor = np.linspace (1,len(x),len(x))
ver = 0.5*m*((x/Difference_Method_Step_size)**2)
plt.figure()
plt.grid()
plt.plot(hor, ver)

print (x[deltaX])
