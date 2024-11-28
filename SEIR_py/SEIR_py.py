from numpy import array as vector
import numpy as np
import matplotlib.pyplot as plot

# Explizites Euler-Verfahren
def euler_method(f, t0, x0, t1, h):
    t = t0; x = x0
    a = [[t, x]]
    for k in range(0, 1 + int((t1 - t0)/h)):
        t = t0 + k*h
        x = x + h*f(t, x)
        a.append([t, x])
    return a

# Implizites Euler-Verfahren
def implicit_euler_method(f, t0, x0, t1, h):
    print("Implizites Euler")
    """
    Implizites Euler-Verfahren zur Lösung von ODEs.
    
    Parameter:
    - f: Funktion f(t, x), die die rechte Seite der ODE definiert (dx/dt = f(t, x)).
    - t0: Anfangszeit.
    - x0: Anfangszustand (Startwert für x).
    - t1: Endzeit.
    - h: Schrittweite.
    
    Rückgabe:
    - t_vals: Liste der Zeitpunkte.
    - x_vals: Liste der Zustände x zu den jeweiligen Zeitpunkten.
    """
    
    # Impliziter Euler-Schritt
    def implicit_euler_step(f, y_old, t, h):
        from scipy.optimize import fsolve
        def F(y_new):
            return y_new - y_old - h * f(t + h, y_new)
    
        # Benutze fsolve, um das nichtlineare Gleichungssystem zu lösen
        y_new = fsolve(F, y_old)
        return y_new
    
    # Initialisieren der Zeit- und Lösungsarrays
    t_vals = np.arange(t0, t1 + h, h)
    num_steps = len(t_vals)
    
    x_vals = np.zeros((num_steps, len(x0)))
    x_vals[0] = x0
    
    a = [[t_vals[0], x_vals[0]]]
    # Numerische Lösung mit implizitem Euler-Verfahren
    for i in range(1, num_steps):
        x_vals[i] = implicit_euler_step(f, x_vals[i-1], t_vals[i-1], h)
        a.append([t_vals[i], x_vals[i]])
    
    return a

# Klassisches Runge-Kutta-Verfahren
def rk4_method(f, t0, x0, t1, h):
    t = t0; x = x0
    a = [[t, x]]
    for k in range(0, 1 + int((t1 - t0)/h)):
        t = t0 + k*h
        k1 = f(t, x)
        k2 = f(t + 0.5*h, x + 0.5*h*k1)
        k3 = f(t + 0.5*h, x + 0.5*h*k2)
        k4 = f(t + h, x + h*k3)
        x = x + h*(k1+2.0*k2+2.0*k3+k4)/6.0
        a.append([t, x])
    return a

# Dormand-Prince-4,5-Verfahren
def dopri45_method(f, t0, x0, t1, h, tol=1e-6):
    """
    Dormand-Prince 4(5)-Methode zur Lösung von ODEs mit adaptiver Schrittweitensteuerung.
    
    Parameter:
    - f: Funktion f(t, x), die die rechte Seite der ODE definiert (dx/dt = f(t, x)).
    - t0: Anfangszeit.
    - x0: Anfangszustand (Startwert für x).
    - t1: Endzeit.
    - h: Anfangsschrittweite.
    - tol: Toleranz für den adaptiven Schrittweitenalgorithmus (Fehlersteuerung).
    
    Rückgabe:

    """
    # Dormand-Prince 4(5) Koeffizienten
    c = [0, 1/5, 3/10, 4/5, 8/9, 1, 1]
    
    a = [
        [],  # a_1 (unused)
        [1/5],  # a_2
        [3/40, 9/40],  # a_3
        [44/45, -56/15, 32/9],  # a_4
        [19372/6561, -25360/2187, 64448/6561, -212/729],  # a_5
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],  # a_6
        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]  # a_7
    ]

    # b-Koeffizienten für die 4. und 5. Ordnung
    b4 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]  # 4. Ordnung
    b5 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 5. Ordnung

    res = [[t0, x0]]

    t = t0
    x = x0

    while t < t1:
        if t + h > t1:
            h = t1 - t

        # Berechnung der k-Werte (k1 bis k7)
        k1 = h * f(t, x)
        k2 = h * f(t + c[1]*h, x + a[1][0]*k1)
        k3 = h * f(t + c[2]*h, x + a[2][0]*k1 + a[2][1]*k2)
        k4 = h * f(t + c[3]*h, x + a[3][0]*k1 + a[3][1]*k2 + a[3][2]*k3)
        k5 = h * f(t + c[4]*h, x + a[4][0]*k1 + a[4][1]*k2 + a[4][2]*k3 + a[4][3]*k4)
        k6 = h * f(t + c[5]*h, x + a[5][0]*k1 + a[5][1]*k2 + a[5][2]*k3 + a[5][3]*k4 + a[5][4]*k5)
        k7 = h * f(t + c[6]*h, x + a[6][0]*k1 + a[6][1]*k2 + a[6][2]*k3 + a[6][3]*k4 + a[6][4]*k5 + a[6][5]*k6)

        # 4. und 5. Ordnung Lösung
        x4 = x + b4[0]*k1 + b4[1]*k2 + b4[2]*k3 + b4[3]*k4 + b4[4]*k5 + b4[5]*k6 + b4[6]*k7
        x5 = x + b5[0]*k1 + b5[1]*k2 + b5[2]*k3 + b5[3]*k4 + b5[4]*k5 + b5[5]*k6 + b5[6]*k7

        # Fehlerabschätzung und Schrittweitenanpassung
        error = np.linalg.norm(x5 - x4)
        if error < tol:
            t += h
            x = x4
            res.append([t, x])

        # Anpassung der Schrittweite
        h = h * min(2, max(0.1, 0.9 * (tol / error) ** 0.2))
    
    return res

def SEIR_model(alpha, beta, gamma):
    def f(t, x):
        s, e, i, r = x
        return vector([
            -beta*s*i,
            beta*s*i - alpha*e,
            alpha*e - gamma*i,
            gamma*i
        ])
    return f

def SEIR_simulation(sim_method, alpha, beta, gamma, e0, i0, days, step=0.8):
    x0 = vector([1.0 - e0 - i0, e0, i0, 0.0])
    f = SEIR_model(alpha, beta, gamma)
    return sim_method(f, 0, x0, days, step)
  
class Simulation:
    def __init__(self):
        self.label= ""
        self.sim= None
        self.t= []
        self.s= []
        self.e= []
        self.i= []
        self.r= []
        
sims= {}

def diagram(simulations):
    figure, axes = plot.subplots(1,1)

    figure.suptitle('SEIR Model')
    plot.style.use('fivethirtyeight')
    
    sim = simulations["Explicit Euler"]
    sim.t,sim.x = zip(*sim.sim())
    sim.s, sim.e, sim.i, sim.r = zip(*sim.x)
    
    axes.grid(linestyle = ':', linewidth = 2.0, color = "#808080")
    axes.set_title(sim.label)
    axes.plot(sim.t, sim.s, color="#0000cc", linewidth = 2, label="S (Susceptible)")
    axes.plot(sim.t, sim.e, color="#ffb000", linestyle='--', linewidth = 2, label="E (Exposed)")
    axes.plot(sim.t, sim.i, color="#a00060", linestyle='-.', linewidth = 2, label="I (Infectious)")
    axes.plot(sim.t, sim.r, color="#008000", linestyle=':', linewidth = 2, label="R (Removed)")
    axes.set_facecolor('white')
    axes.legend()
        
    plot.show()
    
def diagrams(simulations):
    figure, axes = plot.subplots(2,len(simulations)//2)

    figure.suptitle('SEIR Model')
    plot.style.use('fivethirtyeight')
    
    idxs = [[0,0], [0,1], [1,0], [1,1]]
    k= 0
    for label, sim in simulations.items():
        i,j= idxs[k]
        k= k+1
        print(sim.label)
        sim.t,sim.x = zip(*sim.sim())
        sim.s, sim.e, sim.i, sim.r = zip(*sim.x)
        figure.subplots_adjust(bottom = 0.15)
        try:
            axes[i,j].grid(linestyle = ':', linewidth = 2.0, color = "#808080")
            axes[i,j].set_title(sim.label)
            axes[i,j].plot(sim.t, sim.s, color="#0000cc", linewidth = 1.5, label="S (Susceptible)")
            axes[i,j].plot(sim.t, sim.e, color="#ffb000", linestyle='--', linewidth = 1.5, label="E (Exposed)")
            axes[i,j].plot(sim.t, sim.i, color="#a00060", linestyle='-.', linewidth = 1.5, label="I (Infectious)")
            axes[i,j].plot(sim.t, sim.r, color="#008000", linestyle=':', linewidth = 1.5, label="R (Removed)")
        except:
            continue
        
    plot.show()
    
def comparison(simulations):
    figure, axes = plot.subplots(2,2)

    figure.suptitle('SEIR Model')
    plot.style.use('fivethirtyeight')
    
    colors = ["#ffb000", "#a00060", "#0000cc", "#008000"]
    linestyles = ["solid", "--", "-.", ":"]
    
    j= 0
    for label, sim in simulations.items():
        print(sim.label)
        sim.t,sim.x = zip(*sim.sim())
        sim.s, sim.e, sim.i, sim.r = zip(*sim.x)
        figure.subplots_adjust(bottom = 0.15)
        axes[0,0].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[0,0].set_facecolor('white')
        axes[0,1].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[0,1].set_facecolor('white')
        axes[1,0].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[1,0].set_facecolor('white')
        axes[1,1].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[1,1].set_facecolor('white')
        axes[0,0].set_title('Exposed')
        axes[0,0].plot(sim.t, sim.e, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[0,0].legend()
        axes[0,1].set_title('Infectious')
        axes[0,1].plot(sim.t, sim.i, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[0,1].legend()
        axes[1,0].set_title('Susceptible')
        axes[1,0].plot(sim.t, sim.s, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[1,0].legend()
        axes[1,1].set_title('Removed')
        axes[1,1].plot(sim.t, sim.r, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[1,1].legend()
        
        j=j+1
        
    plot.show()



def simulation1():
    global N, R0, gamma, e0, i0, days
    return SEIR_simulation(euler_method,
        alpha = 1/5.5, beta = R0*gamma, gamma = gamma,
        e0 = e0, i0 = i0, days = days)

def simulation2():
    global N, R0, gamma, e0, i0, days
    return SEIR_simulation(implicit_euler_method,
        alpha = 1/5.5, beta = R0*gamma, gamma = gamma,
        e0 = e0, i0 = i0, days = days)

def simulation3():
    global N, R0, gamma, e0, i0, days
    return SEIR_simulation(rk4_method,
        alpha = 1/5.5, beta = R0*gamma, gamma = gamma,
        e0 = e0, i0 = i0, days = days)

def simulation4():
    global N, R0, gamma, e0, i0, days
    return SEIR_simulation(dopri45_method,
        alpha = 1/5.5, beta = R0*gamma, gamma = gamma,
        e0 = e0, i0 = i0, days = days)

N = 83200000 # Einwohnerzahl von Deutschland 2019/2020
R0 = 4.8; gamma = 1/3.0

e0, i0, days = 40000.0/N,  10000.0/N, 140
e0, i0, days = 40000.0/N,  10000.0/N, 80 

sim1, sim2, sim3, sim4 = Simulation(), Simulation(), Simulation(), Simulation()
sim1.label, sim1.sim = "Explicit Euler", simulation1
sim2.label, sim2.sim = "Implicit Euler", simulation2
sim3.label, sim3.sim = "Runge-Kutta", simulation3
# sim4.label, sim4.sim = "Dormand-Prince 45", simulation4

sims= {}
sims[sim1.label]= sim1
sims[sim2.label]= sim2
sims[sim3.label]= sim3
# sims[sim4.label]= sim4

# diagrams(sims)
comparison(sims)

diagram(sims)