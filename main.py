from flask import Flask, request, jsonify
from functions.gauss_seidel import Gauss_s
from functions.lineal_equations import eliminacion_gaussiana
from functions.ceros import biseccion, pos_falsa, newton, secante
from functions.transform import cast_to_function
import numpy as np
import sympy as sp


app = Flask(__name__)

# Ceros ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route("/ceros-biseccion", methods=["POST"])
def ceros_biseccion_solution():
    data = request.get_json()

    # Convertir la expresión en texto a una función
    f = cast_to_function(str(data["function"]))

    # Definir los límites del intervalo y la tolerancia desde el body
    a = int(data["lim_inferior"])
    b = int(data["lim_superior"])
    tolerancia = int(data["tolerancia"])

    # Aplicar el método de bisección
    try:
        raiz, iteraciones, valores_iteracion = biseccion(f, a, b, tolerancia)

        response = {
            "raiz": raiz,
            "iteraciones": iteraciones,
            "valores_iteracion": valores_iteracion
        }

        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    


@app.route("/ceros-falsa-posicion", methods=["POST"])
def ceros_falsa_pos_solution():
    data = request.get_json()

    f = cast_to_function(str(data["function"]))
    a = int(data["lim_inferior"])
    b = int(data["lim_superior"])
    tolerancia = float(data["tolerancia"])  # Asegúrate de que sea un float

    raiz, iteraciones, valores_iteracion = pos_falsa(f, a, b, tolerancia)
    
    if raiz is None:
        return jsonify({"error": "El intervalo no cumple con el teorema de Bolzano (f(a) * f(b) >= 0)"}), 400

    response = {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "valores_iteracion": valores_iteracion
    }
    return jsonify(response), 200


@app.route("/ceros-newton", methods=["POST"])
def ceros_newton_solution():
    data = request.get_json()

    expr_text = str(data["function"])
    expr = sp.sympify(expr_text)
    a = float(data["lim_inferior"])
    b = float(data["lim_superior"])
    seed = (a + b) / 2

    tolerancia = float(data["tolerancia"])  # Asegúrate de que sea un float

    raiz, iteraciones, valores_iteracion = newton(expr, seed, tolerancia)

    response = {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "valores_iteracion": valores_iteracion
    }
    return jsonify(response), 200


@app.route("/ceros-secante", methods=["POST"])
def ceros_secante_solution():
    data = request.get_json()

    expr_text = str(data["function"])
    expr = sp.sympify(expr_text)
    a = float(data["lim_inferior"])
    b = float(data["lim_superior"])

    tolerancia = float(data["tolerancia"])  # Asegúrate de que sea un float

    raiz, iteraciones, valores_iteracion = secante(expr, a, b, tolerancia)

    response = {
        "raiz": raiz,
        "iteraciones": iteraciones,
        "valores_iteracion": valores_iteracion
    }
    return jsonify(response), 200
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route("/ecuaciones-lineales-eliminacion-g", methods=["POST"])
def lineal_eq_gauss_elimination_solution():
    data = request.get_json()

    matrix_A = np.array(data["matrix_A"])
    matrix_B = np.array(data["matrix_B"])

    result = eliminacion_gaussiana(matrix_A, matrix_B)

    response = {
        "result": result.tolist(),
    }
    return jsonify(response), 200

# Lineal Equations ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



@app.route("/gauss-seidel", methods=["POST"])
def gauss_seidel_solution():
    A = [[4, -1, 0, -1, 0, 0, 0, 0, 0],
         [-1, 4, -1, 0, -1, 0, 0, 0, 0],
         [0, -1, 4, 0, 0, -1, 0, 0, 0],
         [-1, 0, 0, 4, -1, 0, -1, 0, 0],
         [0, -1, 0, -1, 4, -1, 0, -1, 0],
         [0, 0, -1, 0, -1, 4, 0, 0, -1],
         [0, 0, 0, -1, 0, 0, 4, -1, 0],
         [0, 0, 0, 0, -1, 0, -1, 4, -1],
         [0, 0, 0, 0, 0, -1, 0, -1, 4]]

    b = [50, 20, 60, 30, 0, 40, 30, 0, 40]
    x0 = np.zeros(9)
    tol = 1e-6

    try:
        x_sol, errores = Gauss_s(A, b, x0, tol)
        
        result = {
            "x_sol": x_sol.tolist(),  # Convertir a lista para que sea JSON serializable
            "errores": errores
        }

        print(errores)
        print("---------------------------separate---------------------------")
        print(x_sol)

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    


if __name__ == "__main__":
    app.run(debug=True)