from flask import Flask, request, jsonify
from functions.gauss_seidel import Gauss_s
from functions.ceros.biseccion import biseccion
from functions.transform import cast_to_function
import numpy as np

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