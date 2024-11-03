import re
import time
from uuid import uuid3
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os

from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, concatenate_audioclips
)
from openai import OpenAI

from Api.cloudinary_uploader import cloudinary_uploader
from python.fontTools.unicodedata import script
from python.moviepy.video.compositing.concatenate import concatenate_videoclips
from python.openai import audio


# from script_to_audio import script_to_audio
class image:
    def __init__(self):
        self.path = None
        self.image_url = None
        self.start_time = None
        self.duration = None
        self.audio_path = None
        self.audio_url = None

    # def __init__(self, path, cloudinary_url, start_time, duration, audio_path):
    #     self.path = path
    #     self.cloudinary_url = cloudinary_url
    #     self.start_time = start_time
    #     self.duration = duration
    #     self.audio_path = audio_path


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

    def escape_latex(self, text):
        # Use regex to detect LaTeX commands and avoid escaping them
        if re.match(r'^\\', text):
            text.replace('\\\\\\\\', '\\\\')
            return text  # Assume it's a LaTeX command, do not escape
        else:
            special_chars = ['#', '$', '%', '&', '~', '_', '^']
            for char in special_chars:
                text.replace('\\\\\\\\', '\\\\')
                text = text.replace(char, '\\' + char)
            return text

    def render_latex_to_image(self, latex_str, output_image='latex_image.png'):
        plt.rcParams.update({
            "text.usetex": True,
            "font.size": 24,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"
        })
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')

        # Do not wrap the LaTeX string in $$...$$ if it contains a display math environment
        if not latex_str.strip().startswith(r'\begin'):
            latex_str = f"$$ {latex_str} $$"  # Wrap in $$...$$ only if not already in a math environment
        latex_str.replace('\\\\\\\\', '\\\\')
        # print(f"Final LaTeX string to render: {latex_str}")

        ax.text(
            0.5, 0.5, latex_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
        try:
            plt.savefig(output_image, bbox_inches='tight', pad_inches=1, dpi=200)
            # print(f"Image saved as {output_image}")
        except Exception as e:
            print(f'Error with LaTeX shown on screen: {e}')
            raise
        plt.close(fig)

    def create_video_from_inputs(self, video_inputs, output_video='final_movie.mp4'):
        if not video_inputs:
            print("Error: video_inputs list is empty.")
            return None

        start_time = time.time()
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
            return (index, audio_file)

        # Use ThreadPoolExecutor to process scripts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all scripts to the executor
            futures = [executor.submit(process_script, vi, i) for i, vi in enumerate(video_inputs)]

            # As each future completes, process the result
            for future in concurrent.futures.as_completed(futures):
                index, audio_file = future.result()
                if audio_file and os.path.exists(audio_file):
                    try:
                        audio_clip = AudioFileClip(audio_file)
                        if n_channels is None:
                            n_channels = audio_clip.nchannels  # Get the number of channels from the first clip
                        audio_clips.append(audio_clip)
                        audio_duration = audio_clip.duration
                        audio_durations.append(audio_duration)

                        start_times.append(current_time)
                        current_time += audio_duration

                        # Add silence after each audio clip except the last one
                        if index < len(video_inputs) - 1:
                            silence_clip = self.make_silence(silence_duration, n_channels=n_channels)
                            audio_clips.append(silence_clip)
                            current_time += silence_duration
                    except Exception as e:
                        print(f"Error processing audio file {audio_file}: {e}")
                else:
                    print(f"Audio file for index {index} is missing or failed to create.")

        # Check if any audio clips were added
        if not audio_clips:
            print("Error: No audio clips were generated.")
            return None

        # Concatenate all audio clips
        print(f"Number of audio clips: {len(audio_clips)}")
        final_audio = concatenate_audioclips(audio_clips)

        # Proceed with the rest of your video creation logic...
        # [Your existing video creation code continues here]

        # Example continuation:
        # ...
        # video_clip = concatenate_videoclips(line_clips)
        # video_clip = video_clip.set_audio(final_audio)
        # video_clip.write_videofile(...)
        # Cleanup files
        # ...

        print(f"Total time taken: {time.time() - start_time} seconds.")
        return output_video
