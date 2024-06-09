import sympy as sp


def cast_to_function(expr_text):
    # Definir la variable simbólica
    x = sp.symbols('x')
    
    # Convertir el texto en una expresión SymPy
    expr = sp.sympify(expr_text)
    
    # Convertir la expresión SymPy en una función lambda
    f = sp.lambdify(x, expr, 'numpy')
    
    return f


def cast_to_function_taylor(expr_text):
    # Definir la variable simbólica
    x = sp.symbols('x')
    
    # Convertir el texto en una expresión SymPy
    expr = sp.sympify(expr_text)
    
    return expr