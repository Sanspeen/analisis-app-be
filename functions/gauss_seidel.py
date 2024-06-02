import numpy as np

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
            #print(f'Iteración: {cont}\nVector: {x1}\nError: {error}')
            xo = x1
            x1 = np.dot(Tg, xo) + Cg
            error = max(np.abs(x1 - xo))
            errores.append(error)
            cont += 1
        
       #print(f'Iteración: {cont}\nVector: {x1}\nError: {error}')
        return x1, errores
    else:
        print('El sistema iterativo no converge a la solución única del sistema')
        return None, errores
