import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

def S_taylor(f, x0, n):
    x = sp.symbols('x')
    P = 0
    coefficients = []
    for k in range(n + 1):
        df = sp.diff(f, x, k)
        dfxo = df.subs(x, x0)
        P = P + dfxo * (x - x0) ** k / factorial(k)
        coefficients.append(float(dfxo) / factorial(k))  # Convertir a float
    return coefficients

def cota_t(f, x0, xp, n):
    x = sp.symbols('x')
    m = min(x0, xp)
    M = max(x0, xp)
    u = np.linspace(m, M, 500)
    df = sp.diff(f, x, n+1)
    df = sp.lambdify(x, df)
    Mc = np.max(abs(df(u)))
    return Mc*np.abs((xp-x0) ** (n+1) / factorial(n+1))