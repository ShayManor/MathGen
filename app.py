import json
import os
import time
import uuid
from enum import Enum

from flask import Flask, request, jsonify
from flask_cors import CORS

from Api.export_file import get_in_bucket
from Movie_Creator.solver import solver  # Adjust the import as per your project structure

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


class problem_type(Enum):
    MATH = True
    WORD = False


def solve(word_or_math: problem_type):
    start_time = time.time()
    data = request.get_json()
    problem = data.get('problem')
    if not problem:
        return jsonify({'error': 'No problem provided'}), 400

    result = solver(problem)
    print(f'Total time E2E: {time.time() - start_time} seconds')
    if word_or_math.value:
        result.solve_math()
    else:
        result.solve_word()
    return result.upload()


@app.route('/')
def index():
    return json.dumps({'url': 'test123'})


@app.route('/math', methods=['POST'])
def math():
    return solve(problem_type.MATH)


@app.route('/word', methods=['POST'])
def word():
    return solve(problem_type.WORD)


@app.route('/ping/uuid_str', methods=['GET'])
def ping(uuid_str):
    return get_in_bucket(uuid_str + 'final_movie.mp4')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
