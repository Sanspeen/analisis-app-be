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


def Gauss_s(A, b, xo, tol):
    cont = 0
    errores = []

    D = np.diag(np.diag(A))
    L = D - np.tril(A)
    U = D - np.triu(A)

    Tg = np.dot(np.linalg.inv(D - L), U)  # (D-L)^-1 U
    Cg = np.dot(np.linalg.inv(D - L), b)  # (D-L)^-1 b

    lam, vec = np.linalg.eig(Tg)  # Calcula los valores propios
    radio = max(abs(lam))  # Calcula el radio espectral
    print(f'Radio espectral de Tg: {radio}')

    if radio < 1:
        x1 = np.dot(Tg, xo) + Cg
        error = max(np.abs(x1 - xo))
        errores.append(error)
        cont += 1

        while error > tol:
            xo = x1
            x1 = np.dot(Tg, xo) + Cg
            error = max(np.abs(x1 - xo))
            errores.append(error)
            cont += 1

        return x1, errores, radio
    else:
        print('El sistema iterativo no converge a la solución única del sistema')
        return None, errores, radio