# Eingebettetes Runge-Kutta-Verfahren

#%%

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Two Body Problem

def F_call(u):
    #return F ausgewertet an der Stelle u
    #G = 6.6743015*(10**(-11)) --> umwandeln in AE *10**(-12)
    #M = #Masse Stern
    G = 6.6738*(10**(-11))*(10**(-12))
    M = 1.989*(10**30)
    r = np.linalg.norm(u[0:2])
    a = -G*M/(r**3)
    A = np.array([[0,0,1,0],[0,0,0,1],[a,0,0,0],[0,a,0,0]])
    
    return A@np.array(u)

#%% Eingebettetes Runge-Kutta-Verfahren, Stufen 2 und 4

def eing_runge_kutta(u0, tau_0, A, b1, b2, nbr_Steps, tol):
    u_tk = u0
    res = []
    res.append(u_tk)
    
    for k in range(1, nbr_Steps):
        tau = tau_0
        f_i = []
        for i in range(4):
            sum = 0
            for j in range(i):
                sum += A[i,j]*f_i[j]
            f_i.append(F_call(u_tk + tau*sum))
        
        sum1 = 0
        for i in range(2):
            sum1 += b1[i]*f_i[i]
        u_tk1_Plus1 = u_tk + tau*sum1
        sum2 = 0
        for i in range(4):
            sum2 += b2[i]*f_i[i]
        u_tk2_Plus1 = u_tk + tau*sum2
        
        while np.linalg.norm(u_tk1_Plus1 - u_tk2_Plus1) > tol:   #!!!would it be better, if I would look 
                                                                #at the differences within the singe components 
                                                                #of the vector instead of looking at the norm 
                                                                #of the difference?
            tau = 0.8 * tau
            
            f_i = []
            for i in range(4):
                sum = 0
                for j in range(i):
                    sum += A[i,j]*f_i[j]
                f_i.append(F_call(u_tk + tau*sum))
            sum1 = 0
            for i in range(2):
                sum1 += b1[i]*f_i[i]
            u_tk1_Plus1 = u_tk + tau*sum1
            sum2 = 0
            for i in range(4):
                sum2 += b2[i]*f_i[i]
            u_tk2_Plus1 = u_tk + tau*sum2
            
        u_tkPlus1 = u_tk2_Plus1
        res.append(u_tkPlus1)
         
        # Prepare for next step
        u_tk = u_tkPlus1
        
    return np.array(res)
            
#%% Visualisation           
            
plt.close('all')

u0 = np.array([1,0,-100*10**(-9),150000000*10**(-4)]) #no idea which values to choose
tau_0 = 1000 #time intervall we choose
nbr_Steps = 5000 #iteration steps
tol = 1e-2

# A und b für Runge-Kutta
A = np.array([[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]])
b1 = np.array([1/2,1/2])
b2 = np.array([1/6,1/3,1/3,1/6])


runge_kutta_res = eing_runge_kutta(u0, tau_0, A, b1, b2, nbr_Steps, tol)

figure, ax = plt.subplots()
x_runge_kutta = runge_kutta_res[:,0]
y_runge_kutta = runge_kutta_res[:,1]

plt.plot(x_runge_kutta, y_runge_kutta, 'midnightblue', label='Runge-Kutta')

x_neg = runge_kutta_res[0,0]
for i in range(1,len(runge_kutta_res)):
    if runge_kutta_res[i,0] < x_neg:
        x_neg = runge_kutta_res[i,0]
x_pos = runge_kutta_res[0,0]
for i in range(1,len(runge_kutta_res)):
    if runge_kutta_res[i,0] > x_pos:
        x_pos = runge_kutta_res[i,0]
        
y_neg = runge_kutta_res[0,1]
for i in range(1,len(runge_kutta_res)):
    if runge_kutta_res[i,1] < y_neg:
        y_neg = runge_kutta_res[i,1]
y_pos = runge_kutta_res[0,1]
for i in range(1,len(runge_kutta_res)):
    if runge_kutta_res[i,1] > y_pos:
        y_pos = runge_kutta_res[i,1]
        
ax.set_xlim(x_neg, x_pos)
ax.set_ylim(y_neg, y_pos)

ax.legend()

plt.show()           
            
            
            
            
            
            
            
            
            