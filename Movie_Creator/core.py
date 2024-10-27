from on_screen_text_generator import on_screen_generator
from process_user_data import process_data
from script_generator import script_generator
from video_input import video_input


class core:
    def __init__(self, problem, api_key):
        self.problem = problem
        self.apiKey = api_key

    def start(self):
        process = process_data(self.problem, api_key=self.apiKey)
        post_processed_data = process.start_processing()
        print("Post Processing Finished")
        print(post_processed_data)
        if post_processed_data == "Error":
            return False

        script = script_generator(apiKey=self.apiKey, prompt=post_processed_data)
        script = script.start_process()
        print("Script finished")

        sliced_script = script.split('~')
        print('Length is ' + str(len(sliced_script)))
        screen_text_obj = on_screen_generator(apiKey=self.apiKey)
        num_steps = len(sliced_script)

        show_on_screen = []

        if num_steps == 1:
            v_input = video_input()
            v_input.script = sliced_script[0]
            v_input.on_screen = screen_text_obj.start_process(block=sliced_script[0], pre_block=None, post_block=None)
            show_on_screen.append(v_input)
        else:
            for i in range(num_steps):
                v_input = video_input()
                v_input.set_script(sliced_script[i])
                if i == 0:
                    v_input.on_screen = screen_text_obj.start_process(block=sliced_script[i], pre_block=None,
                                                                      post_block=sliced_script[i + 1])
                elif i == num_steps - 1 and i > 0:
                    v_input.on_screen = screen_text_obj.start_process(block=sliced_script[i],
                                                                      pre_block=sliced_script[i - 1],
                                                                      post_block=None)
                else:
                    v_input.on_screen = screen_text_obj.start_process(block=sliced_script[i],
                                                                      pre_block=sliced_script[i - 1],
                                                                      post_block=sliced_script[i + 1])
                show_on_screen.append(v_input)
        return show_on_screen
