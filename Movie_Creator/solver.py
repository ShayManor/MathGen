import json
import os
import time

import dotenv
import openai

from Api.export_file import upload_to_bucket
from Movie_Assembler.create_movie import create_movie
from Movie_Creator.core import core


class solver:
    def __init__(self, problem):
        self.movie_name = None
        self.name = None
        self.problem = problem
        self.link = ''
        self.api_key = dotenv.get_key('.env', 'OPENAI_API_KEY')

    def solve_math(self):
        self.name = self.problem + '.mp4'
        core_instance = core(self.problem, self.api_key)
        video_inputs = core_instance.math_start()
        mov = create_movie(self.api_key)
        self.movie_name = mov.create_video_from_inputs(video_inputs=video_inputs)

    def solve_word(self):
        self.name = self.problem + '.mp4'
        core_instance = core(self.problem, self.api_key)
        video_inputs = core_instance.word_start()
        mov = create_movie(self.api_key)
        self.movie_name = mov.create_video_from_inputs(video_inputs=video_inputs)

    def upload(self):
        self.link = upload_to_bucket(path_to_file=self.movie_name, object_name=self.name.replace(' ', ''))
        return self.to_json()

    def to_json(self):
        return json.dumps({'url': self.link})
