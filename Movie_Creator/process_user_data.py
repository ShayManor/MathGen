from openai import OpenAI


class process_data:
    def __init__(self, problem, api_key):
        OpenAI.api_key = api_key
        self.assistant_id = "asst_VRHhu2ImFPxGBi05iz2AnjOc"
        self.client = OpenAI()
        self.prompt = "Problem: " + problem

    def start_processing(self):
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
            if ai_response == "false":
                return "Error"
            return ai_response
        else:
            return "Error"
