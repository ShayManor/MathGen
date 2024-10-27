from flask import Flask, request, jsonify
from flask_cors import CORS

from Movie_Creator.solver import solver  # Adjust the import as per your project structure

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
def index():
    return "HelloWorld2!"


@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    problem = data.get('problem')
    result = solver(problem).upload()
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
