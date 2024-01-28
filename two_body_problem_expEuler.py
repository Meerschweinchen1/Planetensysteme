'''
Idee: Verschiedene numerische Verfahren benutzen, um die Lösung des Zweikörperproblems zu approximieren.
- Explizites Eulerverfahren
- Implizites Eulerverfahren
- Crank-Nicolson-Verfahren
- Runge-Kutta-Verfahren
'''
#%%

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
    A = np.array([[0,0,1,0],[0,0,0,1],[a,0,0,0],[0,a,0,0]])
    
    return np.array([u[2],u[3],a*u[0],a*u[1]])

#%% Explizites Eulerverfahren

def expliciteEuler(u0, tau, T):
    T *= (365.2425)
    t = 0
    res = []
    u_tk = u0
    
    res.append(u_tk) # Step for tau = 0
    
    while t < T:
        F_tk = F_call(u_tk)
        Ftk = F_tk * tau
        u_tkPlus1 = (u_tk + Ftk)
        
        # Prepare for next step 
        u_tk = u_tkPlus1
        res.append(u_tkPlus1)
        t += tau
          
    return np.array(res)

#%% Visualisation

plt.close('all')

u0 = np.array([1,0,0,21*24*60*60*(10**(-8))])
tau = 1 #time intervall we choose
T = 6 #iteration steps


expEuler_res = expliciteEuler(u0, tau, T)

figure, ax = plt.subplots()
x_explEuler = expEuler_res[:,0]
y_explEuler = expEuler_res[:,1]

plt.plot(x_explEuler, y_explEuler, 'b', label='explicite Euler')

x_neg = min(expEuler_res[:,0])
x_pos = max(expEuler_res[:,0])
y_neg = min(expEuler_res[:,1])
y_pos = max(expEuler_res[:,1])
        
ax.set_xlim(x_neg, x_pos)
ax.set_ylim(y_neg, y_pos)

plt.show()
