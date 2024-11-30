import json
import os
import time

import openai

from Api.export_file import upload_to_bucket
from Movie_Assembler.create_movie import create_movie
from Movie_Creator.core import core


class solver:
    def __init__(self, problem):
        self.problem = problem
        self.link = ''

    def upload(self, problem: str):
        start_time = time.time()
        name = problem + '.mp4'
        core_instance = core(self.problem, openai.api_key)
        video_inputs = core_instance.start()
        mov = create_movie(openai.api_key)
        movie_name = mov.create_video_from_inputs(video_inputs=video_inputs)
        self.link = upload_to_bucket(path_to_file=movie_name, object_name=name.replace(' ', ''))
        print(f"Final time: {time.time()-start_time}")
        return self.to_json()

    def to_json(self):
        return json.dumps({'url': self.link})
