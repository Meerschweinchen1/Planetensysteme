
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

figure, ax = plt.subplots()
plt.tight_layout(pad=4)

line_expEuler, = plt.plot(expliciteEuler(u0, tau, nbr_Steps)[:,0], expliciteEuler(u0, tau, nbr_Steps)[:,1], 'b', label='explicite Euler')
line_runge_kutta, = plt.plot(runge_kutta(u0, tau, A, b, nbr_Steps)[:,0], runge_kutta(u0, tau, A, b, nbr_Steps)[:,1], 'c', label='Runge-Kutta')

axtau = figure.add_axes([0.15,0.04, 0.75, 0.03])
axsteps = figure.add_axes([0.15,0.01, 0.75, 0.03])

tau_slider = Slider(
    ax = axtau,
    label = 'Schrittweite',
    valmin = 10,
    valmax = 5000,
    valinit = 1000,
    valstep = 10
)

steps_slider = Slider(
    ax = axsteps,
    label = 'Schrittzahl',
    valmin = 1000,
    valmax = 7000,
    valinit = 1000,
    valstep = 500
)

def update(val):
    tau = int(tau_slider.val)
    nbr_Steps = int(steps_slider.val)
    
    ax.clear()
    ax.set_title('Zweikörperproblem')
    
    expEuler_res = expliciteEuler(u0, tau, nbr_Steps)
    runge_kutta_res = runge_kutta(u0, tau, A, b, nbr_Steps)
    
    # set axes limits    
    x_neg = min(np.min(expEuler_res[:,0]), np.min(runge_kutta_res[:,0]))
    x_pos = max(np.max(expEuler_res[:,0]), np.max(runge_kutta_res[:,0]))
    y_neg = min(np.min(expEuler_res[:,1]), np.min(runge_kutta_res[:,1]))
    y_pos = max(np.max(expEuler_res[:,1]), np.max(runge_kutta_res[:,1]))
    
    ax.set_xlim(x_neg - 10, x_pos + 10)
    ax.set_ylim(y_neg - 10, y_pos + 10)
    
    ax.plot(expEuler_res[:,0], expEuler_res[:,1], 'b', label='explicite Euler')
    ax.plot(runge_kutta_res[:,0], runge_kutta_res[:,1], 'c', label='Runge-Kutta')
    figure.canvas.draw_idle()

update(0)
tau_slider.on_changed(update)
steps_slider.on_changed(update)
plt.show()

