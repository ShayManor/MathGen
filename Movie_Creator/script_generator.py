from openai import OpenAI


class script_generator:
    def __init__(self, apiKey, prompt):
        self.client = OpenAI()
        self.client.api_key = apiKey
        self.assistant_id = 'asst_8aFcQaMPRxXJbiGyfVRVDxpJ'
        self.prompt = prompt

    def start_process(self):
        assistant = self.client.beta.assistants.retrieve(
            assistant_id=self.assistant_id
        )
        thread = self.client.beta.threads.create(
            messages=[{"role": "user", "content": self.prompt}]
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        if run.status == 'completed':
            messages = self.client.beta.threads.messages.list(thread_id=run.thread_id)
            ai_response = messages.data[0].content[0].text.value
            return ai_response
        else:
            return "Error"
