from openai import OpenAI
from typing import *

class on_screen_generator:
    def __init__(self):
        self.client = OpenAI()
        # self.assistant_id = 'asst_CoRUsRi8KqrdnZIP4UyCi2WY'
        self.assistant_id = 'asst_x9o2SLVQzX6YKhLVjOtPWHAy'

    def start_process(self, block: str):
        script = block.split('~')
        l = len(script)
        newscr = []
        for i, s in enumerate(script):
            newscr += f'Step {i + 1} of script: {s} \n'

        prompt = f'Number of sections should be: {l}\n Script: {newscr}'
        # prompt = "Script before block: " + (pre_block or "") + "\nBlock script: " + block + "\nScript after block: " + (post_block or "")
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
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            ai_response = messages.data[0].content[0].text.value
            ai_response.replace("\\\\", "\\")
            ai_response.replace("\\\\", "\\")
            split_str = ai_response.split('\n')
            try:
                split_str.remove('```latex')
                split_str.remove('```')
            except:
                print("Fuck you")
            r = ""
            for s in split_str:
                r = r + s + "\n"
            print(r)
            r = r.replace('```latex', '').replace('```', '').strip()
            return r
        else:
            return "Error"
