from flask import Flask, request, jsonify
from gauss_seidel import Gauss_s
import numpy as np

app = Flask(__name__)

# @app.route("/get-user/<user_id>")
# def get_user(user_id):
#     user_data = {
#         "user_id": user_id,
#         "name": "Jhon Doe"
#     }

#     extra = request.args.get("extra")

#     if extra:
#         user_data["extra"] = extra

#     return jsonify(user_data), 200


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