import json
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from Movie_Creator.solver import solver  # Adjust the import as per your project structure

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
def index():
    return json.dumps({'url': 'test123'})


@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json()
    problem = data.get('problem')
    result = solver(problem).upload()
    return result



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
