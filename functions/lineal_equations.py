import numpy as np

def eliminacion_gaussiana(A,b):
    #------------------------
    #A: matriz A
    #b: matriz b
    #------------------------
    n = len(b)
    #se declara el arreglo de los valores de x
    x = np.zeros(n)

    for k in range(0,n-1): #llega hasta n-1 porque debajo del ultimo elemento no hay nada, no tiene que ser 0
        #print(k)
        for i in range(k+1,n):
            #Se encuentra el factor (lambda)
            lam = A[i,k]/(A[k,k])
            #se actualiza la fila
            A[i,k:n]=A[i,k:n]-lam*A[k,k:n]
            b[i]=b[i]-lam*b[k]

    for k in range(n-1, -1, -1):
        x[k]=(b[k]-np.dot(A[k,k+1:n],x[k+1:n]))/(A[k,k])

    #print('x=',x[0],'y=',x[1],'z=',x[2])
    return x