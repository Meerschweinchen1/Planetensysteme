
import numpy as np
import matplotlib.pyplot as plt

#%% Two Body Problem

def F_call(u):
    #return F ausgewertet an der Stelle u
    G = 6.6738*(10**(-11)) # Gravitationskonstante
    M = 1.989*(10**30) # Masse Stern
    r = np.linalg.norm(u[0:2])
    a = -G*M/(r**3)
    A = np.array([[0,0,1,0],[0,0,0,1],[a,0,0,0],[0,a,0,0]])
    
    return A@np.array(u)

#%% Explizites Eulerverfahren

def expliciteEuler(u0, tau, nbr_Steps):
    res = []
    u_tk = u0
    
    res.append(u_tk) # Step for tau = 0
    
    for t in range(1, nbr_Steps+1):
        F_tk = F_call(u_tk)
        Ftk = F_tk * tau
        u_tkPlus1 = (u_tk + Ftk)
        
        # Prepare for next step 
        u_tk = u_tkPlus1
        res.append(u_tkPlus1)
          
    return np.array(res)

#%% Runge-Kutta-Verfahren der Stufe 4

def runge_kutta(u0, tau, A, b, nbr_Steps):
    u_tk = u0
    res = []
    res.append(u_tk)
    
    for k in range(1, nbr_Steps):
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
    return np.array(res)

#%% Visualisation

plt.close('all')

u0 = np.array([1.4960*(10**11),0,-250,200]) 
tau = 1000 #time intervall we choose
nbr_Steps = 2000 #iteration steps

# A und b für Runge-Kutta
A = np.array([[0,0,0,0],[1/2,0,0,0],[0,1/2,0,0],[0,0,1,0]])
b = np.array([1/6,1/3,1/3,1/6])

expEuler_res = expliciteEuler(u0, tau, nbr_Steps)
runge_kutta_res = runge_kutta(u0, tau, A, b, nbr_Steps)

figure, ax = plt.subplots()
x_explEuler = expEuler_res[:,0]
y_explEuler = expEuler_res[:,1]
x_runge_kutta = runge_kutta_res[:,0]
y_runge_kutta = runge_kutta_res[:,1]

plt.plot(x_explEuler, y_explEuler, 'b', label='explicite Euler')
plt.plot(x_runge_kutta, y_runge_kutta, 'c', label='Runge-Kutta')

# set axes limits
x_neg = expEuler_res[0,0]
for i in range(1,len(expEuler_res)):
    if expEuler_res[i,0] < x_neg:
        x_neg = expEuler_res[i,0]
x_pos = expEuler_res[0,0]
for i in range(1,len(expEuler_res)):
    if expEuler_res[i,0] > x_pos:
        x_pos = expEuler_res[i,0]
        
y_neg = expEuler_res[0,1]
for i in range(1,len(expEuler_res)):
    if expEuler_res[i,1] < y_neg:
        y_neg = expEuler_res[i,1]
y_pos = expEuler_res[0,1]
for i in range(1,len(expEuler_res)):
    if expEuler_res[i,1] > y_pos:
        y_pos = expEuler_res[i,1]
 
for i in range(len(runge_kutta_res)):
    if runge_kutta_res[i,0] < x_neg:
        x_neg = runge_kutta_res[i,0]
    if runge_kutta_res[i,0] > x_pos:
        x_pos = runge_kutta_res[i,0]
    if runge_kutta_res[i,1] < y_neg:
        y_neg = runge_kutta_res[i,1]
    if runge_kutta_res[i,1] > y_pos:
        y_pos = runge_kutta_res[i,1]

ax.set_xlim(x_neg - 10, x_pos + 10)
ax.set_ylim(y_neg - 10, y_pos + 10)



ax.legend()

plt.show()





































