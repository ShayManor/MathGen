from PIL.ImageChops import screen

from Movie_Creator.on_screen_text_generator import on_screen_generator
from Movie_Creator.process_user_data import process_data, Assistant
from Movie_Creator.script_generator import script_generator
from Movie_Creator.video_input import video_input


class core:
    def __init__(self, problem, api_key):
        self.problem = problem
        self.apiKey = api_key

    def word_start(self):
        process = process_data(self.problem, api_key=self.apiKey)
        post_processed_data = process.start_processing(Assistant.WORD)
        print("Post Processing Finished")
        return self.start(post_processed_data)

    def math_start(self):
        process = process_data(self.problem, api_key=self.apiKey)
        post_processed_data = process.start_processing(Assistant.MATH)
        print("Post Processing Finished")
        return self.start(post_processed_data)

    def start(self, post_processed_data: str):
        # if post_processed_data == "Error":
        # return False

        script = script_generator(apiKey=self.apiKey, prompt=post_processed_data)
        script = script.start_process()
        print("Script finished")

        sliced_script = script.split('~')
        print('Length is ' + str(len(sliced_script)))
        screen_text_obj = on_screen_generator()
        num_steps = len(sliced_script)
        show_on_screen = []
        on_screen = screen_text_obj.start_process(script).split('\n')
        for i in range(num_steps):
            vi = video_input()
            vi.on_screen = on_screen[i]
            vi.script = sliced_script[i]
            show_on_screen.append(vi)
        # if num_steps == 1:
        #     v_input = video_input()
        #     v_input.script = sliced_script[0]
        #     v_input.on_screen = screen_text_obj.start_process(block=sliced_script[0])
        #     show_on_screen.append(v_input)
        # else:
        #     for i in range(num_steps):
        #         v_input = video_input()
        #         v_input.set_script(sliced_script[i])
        #         v_input.on_screen = screen_text_obj.start_process(sliced_script[i])
        #         show_on_screen.append(v_input)
        # else:
        #     for i in range(num_steps):
        #         v_input = video_input()
        #         v_input.set_script(sliced_script[i])
        #         if i == 0:
        #             v_input.on_screen = screen_text_obj.start_process(block=sliced_script[i], pre_block=None,
        #                                                               post_block=sliced_script[i + 1])
        #         elif i == num_steps - 1 and i > 0:
        #             v_input.on_screen = screen_text_obj.start_process(block=sliced_script[i],
        #                                                               pre_block=sliced_script[i - 1],
        #                                                               post_block=None)
        #         else:
        #             v_input.on_screen = screen_text_obj.start_process(block=sliced_script[i],
        #                                                               pre_block=sliced_script[i - 1],
        #                                                               post_block=sliced_script[i + 1])
        #         show_on_screen.append(v_input)
        return show_on_screen
