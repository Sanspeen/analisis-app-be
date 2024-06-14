import sympy as sp
import re


def cast_to_function(expr_text):
    x = sp.symbols('x')
    expr = sp.sympify(expr_text)
    f = sp.lambdify(x, expr, 'numpy')
    return f

def cast_to_function_diff_eq(func_str):
    # Asegurar que todas las variables estén correctamente formateadas para sympy
    func_str = func_str.replace(' ', '')  # Quitar espacios innecesarios
    func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)  # Asegurar multiplicación implícita
    func_str = re.sub(r'(\d)(\()', r'\1*\2', func_str)  # Multiplicación implícita antes de paréntesis
    func_str = re.sub(r'(\))([a-zA-Z])', r'\1*\2', func_str)  # Multiplicación implícita después de paréntesis

    # Convertir el texto en una expresión SymPy
    try:
        expr = sp.sympify(func_str)
    except sp.SympifyError as e:
        raise ValueError(f"Error al parsear la función: {e}")

    # Convertir la expresión SymPy en una función lambda
    t, y = sp.symbols('t y')
    f = sp.lambdify((t, y), expr, 'numpy')
    return f

def cast_to_function_taylor(expr_text):
    x = sp.symbols('x')
    expr = sp.sympify(expr_text)
    return expr