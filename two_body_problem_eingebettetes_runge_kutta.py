# Eingebettetes Runge-Kutta-Verfahren

#%%

import numpy as np
import matplotlib.pyplot as plt
import time

#%% Two Body Problem

def F_call(u):
    #return F ausgewertet an der Stelle u
    G = 2.959795663*(10**(-4))
    M = 1
    r = np.linalg.norm(u[0:2])
    a = -G*M/(r**3)

    return np.array([u[2],u[3],a*u[0],a*u[1]])

#%% Eingebettetes Runge-Kutta-Verfahren, Stufen 2 und 4

def eing_runge_kutta(u0, tau_0, A, b1, b2, time, tol):
    time = time * (365.2425)
    u_tk = u0
    res = []
    res.append(u_tk)
    
    t = 0
    steps = 0
    
    while t < time:
        tau = tau_0
        f_i = []
        for i in range(4):
            sum = 0
            for j in range(i):
                sum += A[i,j]*f_i[j]
            f_i.append(F_call(u_tk + tau*sum))
        
        sum1 = 0
        sum2 = 0
        for i in range(4):
            sum1 += b1[i]*f_i[i]
            sum2 += b2[i]*f_i[i]
        u_tk1_Plus1 = u_tk + tau*sum1
        u_tk2_Plus1 = u_tk + tau*sum2
        
        while np.linalg.norm(u_tk1_Plus1 - u_tk2_Plus1) > tol:
            steps += 1
            tau = 0.8 * tau
            
            f_i = []
            for i in range(4):
                sum = 0
                for j in range(i):
                    sum += A[i,j]*f_i[j]
                f_i.append(F_call(u_tk + tau*sum))
            sum1 = 0
            sum2 = 0
            for i in range(4):
                sum1 += b1[i]*f_i[i]
                sum2 += b2[i]*f_i[i]
            u_tk1_Plus1 = u_tk + tau*sum1
            u_tk2_Plus1 = u_tk + tau*sum2
        
        u_tkPlus1 = u_tk2_Plus1
        res.append(u_tkPlus1)
         
        # Prepare for next step
        u_tk = u_tkPlus1
        t += tau 
    
    print(steps)
    return np.array(res)
            
#%% Visualisation           
            
plt.close('all')

u0 = np.array([1,0,0,21*24*60*60*(10**(-8))])
tau_0 = 100 #time intervall we choose in days
Time = 10 #whole time intervall in years
tol = 1e-5

# A und b für Runge-Kutta
A = np.array([[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]])
b1 = np.array([1/2,1/2,0,0])
b2 = np.array([1/6,1/3,1/3,1/6])

#start = time.time()
runge_kutta_res = eing_runge_kutta(u0, tau_0, A, b1, b2, Time, tol)
#end = time.time()
#print(end-start)

figure, ax = plt.subplots()
plt.xlabel('AE')
plt.ylabel('AE')

x_runge_kutta = runge_kutta_res[:,0]
y_runge_kutta = runge_kutta_res[:,1]

plt.plot(x_runge_kutta, y_runge_kutta, 'midnightblue', label='eingebettetes Runge-Kutta')

x_neg = min(runge_kutta_res[:,0])
x_pos = max(runge_kutta_res[:,0])
y_neg = min(runge_kutta_res[:,1])
y_pos = max(runge_kutta_res[:,1])
        
ax.set_xlim(x_neg, x_pos)
ax.set_ylim(y_neg, y_pos)

ax.legend()

plt.show()           
            
           
