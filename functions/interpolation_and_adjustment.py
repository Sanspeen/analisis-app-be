import numpy as np


#devuelve un arreglo con los coeficientes del polinomio
def Pol_simple(x_data, y_data):
    #---------------------
    #x_data: la variable independiente
    #y_data: la variable dependiente
    #--------------------
    n=len(x_data)
    xo=np.zeros(n)
    M_p=np.zeros([n,n])
    for i in range(n):
        M_p[i,0]=1
        for j in range(1,n):
            M_p[i,j]=M_p[i,j-1]*x_data[i]
    #a_i=Gauss_s(M_p,y_data,xo,1e-6)
    a_i = np.linalg.solve(M_p, y_data)
    return a_i


#-----------------------
#Lo que se le debe pasar a la función Polu luego de calcular los coeficientes

#a_i=Pol_simple(Px,Ty) 
#ux=np.linspace(min(Px),max(Px),1000)
#--------------------------


#construye el polinomio con los coeficientes hallados
def Poly(a_i,ux):
    P=0
    for i in range(len(a_i)):
        P=P+a_i[i]*ux**i
        
    return P