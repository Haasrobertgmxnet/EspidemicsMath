
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
    from scipy.optimize import fsolve
    def implicit_euler_step(f, y_old, t, h):
        def F(y_new):
            return y_new - y_old - h * f(t + h, y_new)
        y_new = fsolve(F, y_old)
        return y_new

    t_vals = np.arange(t0, t1 + h, h)
    num_steps = len(t_vals)

    x_vals = np.zeros((num_steps, len(x0)))
    x_vals[0] = x0

    a = [[t_vals[0], x_vals[0]]]
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

# SIR-Modell
def SIR_model(beta, gamma):
    def f(t, x):
        s, i, r = x
        return vector([
            -beta*s*i,      # dS/dt
            beta*s*i - gamma*i,  # dI/dt
            gamma*i         # dR/dt
        ])
    return f

# SIR-Simulation
def SIR_simulation(sim_method, beta, gamma, i0, days, step=0.8):
    x0 = vector([1.0 - i0, i0, 0.0])  # Anfangswerte: S, I, R
    f = SIR_model(beta, gamma)
    return sim_method(f, 0, x0, days, step)

# Diagrammfunktionen
def diagram(simulations):
    figure, axes = plot.subplots(1,1)

    figure.suptitle('SIR Model')
    plot.style.use('fivethirtyeight')
    
    sim = simulations["Explicit Euler"]
    sim.t,sim.x = zip(*sim.sim())
    sim.s, sim.i, sim.r = zip(*sim.x)
    
    axes.grid(linestyle = ':', linewidth = 2.0, color = "#808080")
    axes.set_title(sim.label)
    axes.plot(sim.t, sim.s, color="#0000cc", linewidth = 2, label="S (Susceptible)")
    axes.plot(sim.t, sim.i, color="#ffb000", linestyle='--', linewidth = 2, label="I (Infectious)")
    axes.plot(sim.t, sim.r, color="#a00060", linestyle='-', linewidth = 2, label="R (Removed)")
    axes.set_facecolor('white')
    axes.legend()
        
    plot.show()
    
def comparison(simulations):
    figure, axes = plot.subplots(2,2)

    figure.suptitle('SIR Model')
    plot.style.use('fivethirtyeight')
    
    colors = ["#ffb000", "#a00060", "#0000cc", "#008000"]
    linestyles = ["solid", "--", "-.", ":"]
    
    j= 0
    for label, sim in simulations.items():
        if label == "Implicit Euler":
            continue
        print(sim.label)
        sim.t,sim.x = zip(*sim.sim())
        sim.s, sim.i, sim.r = zip(*sim.x)
        figure.subplots_adjust(bottom = 0.15)
        axes[0,0].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[0,0].set_facecolor('white')
        axes[0,1].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[0,1].set_facecolor('white')
        axes[1,0].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[1,0].set_facecolor('white')
        axes[1,1].grid(linestyle = ':', linewidth = 1.0, color = "#808080")
        axes[1,1].set_facecolor('white')
        axes[0,1].set_title('Infectious')
        axes[0,1].plot(sim.t, sim.i, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[0,1].legend()
        axes[1,0].set_title('Susceptible')
        axes[1,0].plot(sim.t, sim.s, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[1,0].legend()
        axes[1,1].set_title('Removed')
        axes[1,1].plot(sim.t, sim.r, color=colors[j], linestyle=linestyles[j], linewidth = 1.5, label=sim.label)
        axes[1,1].legend()

        print("Final Values")
        print('Susceptible', round(sim.s[-1],4), round(83200000*sim.s[-1],2))
        print('Infectious', round(sim.i[-1],4), round(83200000*sim.i[-1],2))
        print('Removed', round(sim.r[-1],4), round(83200000*sim.r[-1],2))
        todos = sim.s[-1] + sim.i[-1] + sim.r[-1]
        print('All', round(todos,4), round(83200000*todos,2))
        print("NEXT")
        
        j=j+1
    
    addPlotAll=True
    
    if addPlotAll:

        sim = simulations["Explicit Euler"]
        sim.t,sim.x = zip(*sim.sim())
        sim.s, sim.i, sim.r = zip(*sim.x)
    
        axes[0,0].grid(linestyle = ':', linewidth = 2.0, color = "#808080")
        axes[0,0].set_title(sim.label)
        axes[0,0].plot(sim.t, sim.s, color="#0000cc", linewidth = 2, label="S (Susceptible)")
        axes[0,0].plot(sim.t, sim.i, color="#ffb000", linestyle='--', linewidth = 2, label="I (Infectious)")
        axes[0,0].plot(sim.t, sim.r, color="#a00060", linestyle='-', linewidth = 2, label="R (Removed)")
        axes[0,0].set_facecolor('white')
        axes[0,0].legend()
        
    plot.show()

# Simulationskonfiguration
class Simulation:
    def __init__(self):
        self.label = ""
        self.sim = None
        self.t = []
        self.s = []
        self.i = []
        self.r = []

sims = {}

# Simulationen erstellen
N = 83200000  # Bevölkerung
R0 = 1.8
gamma = 1/3.0
beta = R0 * gamma
i0 = 10000.0 / N  # Initiale Infizierte
days = 80

sim1 = Simulation()
sim1.label = "Explicit Euler"
sim1.sim = lambda: SIR_simulation(euler_method, beta, gamma, i0, days)

sim2 = Simulation()
sim2.label = "Implicit Euler"
sim2.sim = lambda: SIR_simulation(implicit_euler_method, beta, gamma, i0, days)

sim3 = Simulation()
sim3.label = "Runge-Kutta"
sim3.sim = lambda: SIR_simulation(rk4_method, beta, gamma, i0, days)

sim4 = Simulation()
sim4.label = "Dormand-Prince-4,5"
sim4.sim = lambda: SIR_simulation(dopri45_method, beta, gamma, i0, days)

sims[sim1.label] = sim1
sims[sim2.label] = sim2
sims[sim3.label] = sim3
sims[sim4.label] = sim4

# Simulationen ausführen und plotten
diagram(sims)

comparison(sims)