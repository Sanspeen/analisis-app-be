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