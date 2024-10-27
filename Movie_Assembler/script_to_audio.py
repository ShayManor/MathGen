from openai import OpenAI


class script_to_audio:
    def __init__(self, api_key):
        self.client = OpenAI()
        self.client.api_key = api_key

    def convert(self, script, index):
        speech_file_path = f'{index:=03}audio.mp3'
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=script,
            speed=1.1
        )
        try:
            response.stream_to_file(speech_file_path)
        except:
            print('')


# script_to_audio('sk-ogo9Le8MsYT5nabudc-A0fpbE2hwWlwBTZcn4rt_WbT3BlbkFJooHsUlxuXbgZk_nzFeBJU3aYwYam-9pjxKS_kmst4A').convert('abc', 1)
