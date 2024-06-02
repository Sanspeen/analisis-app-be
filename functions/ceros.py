import numpy as np
import sympy as sp

x = sp.symbols("x")

def biseccion(f, a, b, tolerancia):
    #------------------
    #f: función a la que se van a hallar los ceros
    #a: límite inferior del intervalo
    #b límite superior del intervalo
    #tolerancia
    #------------------
    contador = 0
    valores_iteracion = []

    if f(a)*f(b)<0:
        while abs(a-b)>tolerancia:
            contador+=1
            c= (a+b)/2
            valores_iteracion.append(c)
            #print(f'iteracion {contador}, x={c}')

            if f(a)*f(c)<0:
                b=c
            else:
                a=c
        return c, contador, valores_iteracion
    else:
        print("No cumple el teorema")

def pos_falsa(f, a, b, tolerancia):
    contador = 0
    valores_iteracion = []

    if f(a) * f(b) < 0:
        while True:
            c = a - (f(a) * (a - b)) / (f(a) - f(b))
            valores_iteracion.append(c)
            contador += 1
            if abs(f(c)) <= tolerancia:
                break
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return c, contador, valores_iteracion
    else:
        return None, 0, []  # Retorna None, 0, [] si no cumple con el teorema
    
def newton(expr, x0, tol):
    #--------------------
    # expr: expresión simbólica
    # x0: semilla
    # tol: tolerancia
    #-------------------
    x = sp.symbols('x')
    f = sp.lambdify(x, expr, 'numpy')
    df = sp.lambdify(x, sp.diff(expr, x), 'numpy')
    
    contador = 0
    valores_iteracion = []

    x1 = x0 - f(x0) / df(x0)
    valores_iteracion.append(x1)
    while abs(x1 - x0) > tol:
        x0 = x1
        x1 = x0 - f(x0) / df(x0)
        valores_iteracion.append(x1)
        contador += 1

    return x1, contador, valores_iteracion
        