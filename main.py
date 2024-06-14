from flask import Flask, request, jsonify
from flask_cors import CORS
from functions.taylor_series import S_taylor, cota_t
from functions.differential_equations import Euler, R_Kutta
from functions.interpolation_and_adjustment import Pol_simple, Poly, min_c
from functions.lineal_equations import eliminacion_gaussiana, Gauss_s
from functions.ceros import biseccion, pos_falsa, newton, secante
from functions.transform import cast_to_function, cast_to_function_taylor, cast_to_function_diff_eq
import numpy as np
import sympy as sp


app = Flask(__name__)
CORS(app) 

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

# Lineal Equations ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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


@app.route("/ecuaciones-lineales-gauss-seidel", methods=["POST"])
def lineal_eq_gauss_seidel_solution():
    data = request.get_json()
    
    matrix_A = np.array(data["matrix_A"])
    matrix_B = np.array(data["matrix_B"])
    x0 = np.zeros(len(matrix_B))  # Inicializar x0 como un vector de ceros
    tol = float(data["tolerance"])  # 1e-6

    try:
        x_sol, errores, radio = Gauss_s(matrix_A, matrix_B, x0, tol)
        
        result = {
            "result": x_sol.tolist(),  # Convertir a lista para que sea JSON serializable
            "errores": errores,
            "radio_espectral_tg": radio
        }

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

# Interpolation and adjustment -----------------------------------------------------------------------------------------------------------------------------------------------

@app.route("/interpolacion-lagrange", methods=["POST"])
def interpolation_lagrange_solution():
    data = request.get_json()
    list_a = np.array(data["list_a"]) #var independiente
    list_b = np.array(data["list_b"])
    list_predict = np.array(data["list_predict"])


    pol_simple_result = Pol_simple(list_a, list_b)

    ux = np.linspace(min(list_a), max(list_a), 1000)

    extended_predict_list = list(range(list_predict[0], list_predict[-1]))
    aprox = [Poly(pol_simple_result, i) for i in extended_predict_list]

    result = Poly(pol_simple_result, ux)

    response = {
        "result_polinomio": result.tolist(),
        "result_aprox": aprox
    }
    return jsonify(response), 200


@app.route("/interpolacion-minimos-cuadrados", methods=["POST"])
def interpolation_mc_solution():
    data = request.get_json()
    list_a = np.array(data["list_a"]) #var independiente
    list_b = np.array(data["list_b"])

    result1, result2 = min_c(list_a, list_b)

    response = {    
        "pendiente": result1.tolist(),
        "intercepto": result2.tolist(),

    }
    return jsonify(response), 200


# Diff Eq -------------------------------------------------------------------------------------------------------------------------------------------------------------------


@app.route("/diff-eq-kutta", methods=["POST"])
def diff_eq_kutta_solution():
    data = request.get_json()

    # Convertir la expresión en texto a una función
    f = cast_to_function(str(data["function"]))

    # Definir los límites del intervalo y la tolerancia desde el body
    a = float(data["lim_inferior"])
    b = float(data["lim_superior"])
    initial_conditions = float(data["init_conditions"][0])
    h = float(data["integration_step"])

    try:
        r1, r2 = R_Kutta(f, a, b, initial_conditions, h)

        response = {
            "tiempo": r1.tolist(),
            "yeu": r2
        }

        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/diff-eq-euler", methods=["POST"])
def diff_eq_euler_solution():
    data = request.get_json()

    # Convertir la expresión en texto a una función
    f = cast_to_function_diff_eq(str(data["function"]))

    # Definir los límites del intervalo y la tolerancia desde el body
    a = float(data["lim_inferior"])
    b = float(data["lim_superior"])
    initial_conditions = float(data["init_conditions"][0])
    h = float(data["integration_step"])

    try:
        r1, r2 = Euler(f, a, b, initial_conditions, h)

        response = {
            "tiempo": r1.tolist(),
            "yeu": r2
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
# Taylor series ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@app.route("/series-taylor", methods=["POST"])
def series_taylor_solution():
    data = request.get_json()
    f = cast_to_function_taylor(str(data["function"]))
    x0 = int(data["x0"])
    n = int(data["num_iteraciones"])

    try:
        P = S_taylor(f, x0, n)
        response = {
            "polinomio": str(P)  # Convertir el polinomio a string para enviarlo como JSON
        }
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    app.run(debug=True)