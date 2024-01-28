
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%% Two Body Problem

def F_call(u):
    #return F ausgewertet an der Stelle u
    G = 2.959795663*(10**(-4))
    M = 1
    r = np.linalg.norm(u[0:2])
    a = -G*M/(r**3)

    return np.array([u[2],u[3],a*u[0],a*u[1]])

#%% Runge-Kutta-Verfahren der Stufe 4

def runge_kutta(u0, tau, A, b, T):
    T *= (365.2425)
    t = 0
    u_tk = u0
    res = []
    res.append(u_tk)
    
    while t < T:
        f_i = []
        for i in range(4):
            sum = 0
            for j in range(i):
                sum += A[i,j]*f_i[j]
            f_i.append(F_call(u_tk + tau*sum))
        sum = 0
        for i in range(4):
            sum += b[i]*f_i[i]
        u_tkPlus1 = u_tk + tau*sum
        
        res.append(u_tkPlus1)
        
        # Prepare for next step
        u_tk = u_tkPlus1
        t += tau
        
    return np.array(res)
#%% Visualisation

plt.close('all')

u0 = np.array([1,0,0,21*24*60*60*(10**(-8))])
tau = 20 #time intervall we choose
T = 20

# A und b für Runge-Kutta
A = np.array([[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]])
b = np.array([1/6,1/3,1/3,1/6])


runge_kutta_res = runge_kutta(u0, tau, A, b, T)

figure, ax = plt.subplots()
x_runge_kutta = runge_kutta_res[:,0]
y_runge_kutta = runge_kutta_res[:,1]

plt.plot(x_runge_kutta, y_runge_kutta, 'midnightblue', label='Runge-Kutta')

x_neg = min(runge_kutta_res[:,0])
x_pos = max(runge_kutta_res[:,0])
y_neg = min(runge_kutta_res[:,1])
y_pos = max(runge_kutta_res[:,1])
        
ax.set_xlim(x_neg, x_pos)
ax.set_ylim(y_neg, y_pos)

ax.legend()

plt.show()
