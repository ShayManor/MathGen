import json
import os
import time
from turtledemo.penrose import start

import openai

from Api.export_file import upload_to_bucket
from Movie_Assembler.create_movie import create_movie
from Movie_Creator.core import core


class solver:
    def __init__(self, problem):
        self.problem = problem
        self.link = ''
        os.environ[
            "OPENAI_API_KEY"] = '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'
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def upload(self):
        start_time = time.time()
        core_instance = core(self.problem, openai.api_key)
        video_inputs = core_instance.start()
        mov = create_movie(openai.api_key)
        movie_name = mov.create_video_from_inputs(video_inputs=video_inputs)
        self.link = upload_to_bucket(path_to_file=movie_name)
        print(f"Final time: {time.time()-start_time}")
        return self.to_json()

    def to_json(self):
        return json.dumps({'url': self.link})
