import re
import tempfile
import time
from uuid import uuid3
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os

from moviepy import *
from openai import OpenAI

from Movie_Creator.effects import TextAnimator


# from script_to_audio import script_to_audio
class image:
    def __init__(self):
        self.path = None
        self.image_url = None
        self.start_time = None
        self.duration = None
        self.audio_path = None
        self.audio_url = None

class script_to_audio:
    def __init__(self, api_key):
        self.client = OpenAI()
        self.client.api_key = api_key

    def convert(self, script: str, index: int):
        speech_file_path = f'{index:03}audio.mp3'
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=script,
            speed=1
        )
        try:
            response.stream_to_file(speech_file_path)
            return speech_file_path
        except:
            print('Error in script generation')


class create_movie:
    def __init__(self, api_key):
        self.api_key = api_key

    def make_silence(self, duration, fps=44100, n_channels=2):
        total_samples = int(duration * fps)
        array = np.zeros((total_samples, n_channels))
        return AudioArrayClip(array, fps=fps)

    def create_animated_text_clip(self, latex_str, duration):
        text_clip = TextClip(
            latex_str,
            font_size=50,
            color='white',
            font='Arial',
            method='latex',
            duration=duration,

        )

        # animated_clip = text_clip.crossfadein(1)  # 1 second fade-in
        return text_clip

    def escape_latex(self, text):
        sanitized_text = self.sanitize_input(text)

        # Dictionary of LaTeX special characters and their escaped equivalents
        special_chars = {
            '#': r'\#',
            '$': r'\$',
            '%': r'\%',
            '&': r'\&',
            '~': r'\textasciitilde{}',
            '_': r'\_',
            # '{': r'\{',
            # '}': r'\}',
            # Do not escape backslashes or carets
        }

        # Compile a regex pattern to match any of the special characters
        pattern = re.compile('|'.join(re.escape(key) for key in special_chars.keys()))

        # Escape special characters using the pattern
        escaped_text = pattern.sub(lambda match: special_chars[match.group()], sanitized_text)

        return escaped_text

    def sanitize_input(self, text):
        text = text.replace('```latex', '').replace('```', '').strip()
        return text


    def render_latex_to_image(self, latex_str, image_filename='latex_image.png'):
        # Define the path to save the image
        images_dir = os.path.join(os.getcwd(), 'images')
        os.makedirs(images_dir, exist_ok=True)
        output_image_path = os.path.join(images_dir, image_filename)

        plt.rcParams.update({
            "text.usetex": True,
            "font.size": 24,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
        })

        # Set figure size and DPI to match video resolution
        fig, ax = plt.subplots(figsize=(12, 2), dpi=100)  # Transparent background
        ax.axis('off')

        # Ensure proper LaTeX environment
        if not (latex_str.strip().startswith(r'\begin{align*}') and latex_str.strip().endswith(r'\end{align*}')):
            latex_str = r'\begin{align*}' + latex_str + r'\end{align*}'

        print(f"Final LaTeX string to render: {latex_str}")

        ax.text(
            0.5, 0.5, latex_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )

        try:
            # Save with transparent background
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=1, dpi=100, transparent=True)
            print(f"LaTeX rendered successfully to {output_image_path}")
        except Exception as e:
            print(f'Error with LaTeX rendering: {e}')
            raise
        finally:
            plt.close(fig)

        # Check if the image was created
        if not os.path.exists(output_image_path):
            print(f"Failed to create image: {output_image_path}")
            return None
        else:
            print(f"Image {output_image_path} exists.")
            return output_image_path

    def create_audio(self, video_inputs):
        silence_duration = 0.25  # 0.25 seconds of silence between audio clips

        audio_clips = []
        audio_durations = []
        start_times = []
        current_time = 0.0

        n_channels = None  # Will be set after the first audio clip is loaded
        print(f'Video inputs length: {len(video_inputs)}')

        # Initialize the audio generator
        audio_generator = script_to_audio(self.api_key)

        # Function to process each script
        def process_script(video_input, index):
            audio_file = audio_generator.convert(video_input.script, index)
            if audio_file and os.path.exists(audio_file):
                try:
                    audio_clip = AudioFileClip(audio_file)
                    audio_duration = audio_clip.duration
                    return (index, audio_clip, audio_duration)
                except Exception as e:
                    print(f"Error processing audio file {audio_file}: {e}")
                    return (index, None, 0)
            else:
                print(f"Audio file for index {index} is missing or failed to create.")
                return (index, None, 0)

        N = len(video_inputs)
        results = [None] * N

        # Use ThreadPoolExecutor to process scripts and audio files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all scripts to the executor and keep track of their indices
            futures = {executor.submit(process_script, vi, i): i for i, vi in enumerate(video_inputs)}

            # As each future completes, store the result at the correct index
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    index, audio_clip, audio_duration = future.result()
                    results[index] = (audio_clip, audio_duration)
                except Exception as e:
                    print(f"Error processing script at index {index}: {e}")
                    results[index] = (None, 0)

        # Process the results in order
        for i in range(N):
            audio_clip, audio_duration = results[i]
            if audio_clip:
                if n_channels is None:
                    n_channels = audio_clip.nchannels  # Get the number of channels from the first clip
                audio_clips.append(audio_clip)
                audio_durations.append(audio_duration)
                start_times.append(current_time)
                current_time += audio_duration

                # Add silence after each audio clip except the last one
                if i < N - 1:
                    silence_clip = self.make_silence(silence_duration, n_channels=n_channels)
                    audio_clips.append(silence_clip)
                    current_time += silence_duration
            else:
                print(f"No audio clip for index {i}")

        # Check if any audio clips were added
        if not audio_clips:
            print("Error: No audio clips were generated.")
            return None

        # Concatenate all audio clips
        print(f"Number of audio clips: {len(audio_clips)}")
        final_audio = concatenate_audioclips(audio_clips)
        return final_audio, start_times, results

    def get_position_function(self, i, start_times, base_y, shift_per_clip, x):
        clip_start_time = start_times[i]

        def position(t):
            if t < clip_start_time:
                # Before the clip starts, position it off-screen
                return (x, -1000)  # Off-screen position
            # Count the number of new clips started since this clip started
            num_new_clips = sum(1 for st in start_times[i + 1:] if st <= t)
            y = base_y + shift_per_clip * num_new_clips
            return (x, y)

        return position

    def create_video(self, video_inputs, start_times, results):
        clip_objects = []

        # Define base position and shift per clip
        base_y = 50  # Starting y-position
        shift_per_clip = 60  # Pixels to shift down per clip
        x = 'center'  # x-position

        # Calculate the final duration based on the last clip's end time
        final_duration = max([st + dur for st, (_, dur) in zip(start_times, results) if dur])

        for i, video_input in enumerate(video_inputs):
            audio_clip, audio_duration = results[i]
            if audio_clip is None:
                continue

            latex_str = video_input.on_screen

            # Escape LaTeX special characters
            escaped_latex_str = self.escape_latex(latex_str)

            # Create TextClip
            text_clip = TextClip(
                font="Arial",
                text=escaped_latex_str,
                font_size=50,
                color='white',
                size=(1920, 1080)
            )

            # Set the duration of the TextClip to extend till the end
            text_clip.duration = final_duration - start_times[i]

            # Define position function
            position_func = self.get_position_function(i, start_times, base_y, shift_per_clip, x)
            text_clip.pos = position_func
            text_clip.set_start = start_times[i]

            clip_objects.append(text_clip)

        return clip_objects

    def create_video_from_inputs(self, video_inputs, output_video='final_movie.mp4'):
        if not video_inputs:
            print("Error: video_inputs list is empty.")
            return None

        start_time_video = time.time()

        final_audio, start_times, results = self.create_audio(video_inputs)

        clip_objects = self.create_video(video_inputs, start_times, results)

        final_duration = final_audio.duration
        background = ColorClip(size=(1920, 1080), color=(200, 200, 200))
        background.duration = final_duration

        video = CompositeVideoClip([background] + clip_objects, size=(1920, 1080))
        video.duration = final_duration

        try:
            video.write_videofile(
                output_video,
                fps=20,
                codec='libx264',
                preset='ultrafast',
                audio_codec='aac',
                threads=4,  # Pretty sure this does nothing
                logger='bar'
            )
            print(f"Video successfully created: {output_video}")
        except Exception as e:
            print(f"Error writing video file: {e}")

        # Clean up image files
        for i in range(len(video_inputs)):
            image_file = f"latex_image_line_{i}.png"
            image_path = os.path.join(os.getcwd(), 'images', image_file)
            if os.path.exists(image_path):
                os.remove(image_path)

        print(f"Total time taken: {time.time() - start_time_video} seconds.")
        return output_video
