import numpy as np

def Euler(f, a, b, co, h):
    n = int((b - a) / h)
    t = np.linspace(a, b, n + 1)
    yeu = [co]
    for i in range(n):
        yeu.append(yeu[i] + h * f(t[i], yeu[i]))
    return t, yeu


# Método de Runge-Kutta de cuarto orden (RK4)
def R_Kutta(f, a, b, co, h):
    n = int((b - a) / h)
    t = np.linspace(a, b, n + 1)
    yeu = [co]
    for i in range(n):
        k1 = h * f(t[i])
        k2 = h * f(t[i] + h / 2)
        k3 = h * f(t[i] + h / 2)
        k4 = h * f(t[i] + h)
        y_next = yeu[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yeu.append(y_next)
    return t, yeu