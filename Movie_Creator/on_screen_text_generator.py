from openai import OpenAI
from typing import *


class on_screen_generator:
    def __init__(self, apiKey):
        self.client = OpenAI()
        self.client.api_key = apiKey
        self.assistant_id = 'asst_CoRUsRi8KqrdnZIP4UyCi2WY'

    def start_process(self, pre_block: Optional[str], block: str, post_block: Optional[str]):
        prompt = "Script before block: " + (pre_block or "") + "\nBlock script: " + block + "\nScript after block: " + (post_block or "")
        assistant = self.client.beta.assistants.retrieve(
            assistant_id=self.assistant_id
        )
        # print("In start process")
        thread = self.client.beta.threads.create(
            messages=[{"role": "user", "content": prompt}]
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        if run.status == 'completed':
            messages = self.client.beta.threads.messages.list(thread_id=run.thread_id)
            ai_response = messages.data[0].content[0].text.value
            ai_response.replace("\\\\", "\\")
            ai_response.replace("\\\\", "\\")
            print(ai_response)
            return ai_response
        else:
            return "Error"
