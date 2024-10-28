import re
import time

import matplotlib.pyplot as plt
import numpy as np
import os

from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    ImageClip, AudioFileClip, CompositeVideoClip, concatenate_audioclips
)
from openai import OpenAI


# from script_to_audio import script_to_audio


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
            return text  # Assume it's a LaTeX command, do not escape
        else:
            special_chars = ['#', '$', '%', '&', '~', '_', '^']
            for char in special_chars:
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

        print(f"Final LaTeX string to render: {latex_str}")

        ax.text(
            0.5, 0.5, latex_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
        try:
            plt.savefig(output_image, bbox_inches='tight', pad_inches=1, dpi=200)
            print(f"Image saved as {output_image}")
        except Exception as e:
            print(f'Error with LaTeX shown on screen: {e}')
            raise
        plt.close(fig)

    def create_video_from_inputs(self, video_inputs, output_video='final_movie.mp4'):
        start_time = time.time()
        silence_duration = 0.25  # 0.25 seconds of silence between audio clips

        audio_clips = []
        audio_durations = []
        start_times = []
        current_time = 0.0

        n_channels = None  # Will be set after the first audio clip is loaded

        # Generate audio clips and calculate start times
        for i, video_input in enumerate(video_inputs):
            audio_file = f"{i:03}audio.mp3"

            # Generate audio for each script
            audio_generator = script_to_audio(self.api_key)
            audio_generator.convert(video_input.script, i)

            audio_clip = AudioFileClip(audio_file)
            if n_channels is None:
                n_channels = audio_clip.nchannels  # Get the number of channels from the first clip
            audio_clips.append(audio_clip)
            audio_duration = audio_clip.duration
            audio_durations.append(audio_duration)
            start_times.append(current_time)
            current_time += audio_duration

            # Add silence after each audio clip except the last one
            if i < len(video_inputs) - 1:
                silence_clip = self.make_silence(silence_duration, n_channels=n_channels)
                audio_clips.append(silence_clip)
                current_time += silence_duration

        # Concatenate all audio clips
        final_audio = concatenate_audioclips(audio_clips)

        # Prepare video clips
        video_width, video_height = 1280, 720  # HD resolution

        # Load and resize background image
        # bg_clip = ImageClip('/HelloWorld/images/background.jpeg').set_duration(final_audio.duration).resize((video_width, video_height))

        # Dictionary to hold line clips with their corresponding lines as keys
        line_clips_dict = {}

        # Keep track of lines currently visible
        visible_lines = []

        for i, video_input in enumerate(video_inputs):
            line_id = f"{i:03}"
            current_line = self.escape_latex(video_input.on_screen)

            # Check if the line is already visible
            if current_line in visible_lines:
                print(f"Line already visible, skipping duplicate: {current_line}")
                continue  # Skip adding duplicate lines

            # Add the line to visible lines
            visible_lines.append(current_line)

            # If more than 3 lines are visible, remove the oldest one
            if len(visible_lines) > 3:
                removed_line = visible_lines.pop(0)
                # Update the end time of the removed line's clip
                if removed_line in line_clips_dict:
                    end_time_line = start_times[i]  # Current time
                    line_clips_dict[removed_line] = line_clips_dict[removed_line].set_end(end_time_line)
                    print(f"Line removed: {removed_line}")

            # Build cumulative text from visible lines
            cumulative_text = r' \\ '.join(visible_lines)
            cumulative_text = r'\begin{align*}' + cumulative_text + r'\end{align*}'

            # Render LaTeX image for the current set of visible lines
            image_file = f"latex_image_{line_id}.png"
            self.render_latex_to_image(cumulative_text, image_file)

            # Create ImageClip for the cumulative image
            line_clip = ImageClip(image_file).set_fps(24)
            line_clip = line_clip.set_position('center')

            # Set the timing for the line to appear and disappear
            start_time_line = start_times[i]
            # The line remains visible until it is removed from visible_lines
            if len(visible_lines) == 3:
                # The line will be visible for the duration of the next 3 audio clips
                if i + 3 < len(start_times):
                    end_time_line = start_times[i + 3]
                else:
                    end_time_line = final_audio.duration
            else:
                # The line remains until the next line is added and causes it to be removed
                if i + 1 < len(start_times):
                    end_time_line = start_times[i + 1]
                else:
                    end_time_line = final_audio.duration

            # Set the start and end times
            line_clip = line_clip.set_start(start_time_line).set_end(end_time_line)

            # Add fade-in and fade-out effects
            # if i == 0 or i == len(video_inputs) - 1:
            #     fade_duration = 0.75  # Duration of fade-in and fade-out
            #     line_clip = line_clip.fadein(fade_duration).fadeout(fade_duration)

            # Store the clip in the dictionary
            line_clips_dict[current_line] = line_clip

        # Get all line clips from the dictionary
        line_clips = list(line_clips_dict.values())

        # Composite all clips together
        video_clip = CompositeVideoClip(line_clips, size=(video_width, video_height)).set_fps(24)

        video_clip = video_clip.set_audio(final_audio)
        print(f"DEBUG: Writing video with fps={video_clip.fps}")  # Should output: 24

        # Write the video file
        video_clip.write_videofile(
            output_video,
            fps=video_clip.fps,
            codec='libx264',
            preset='medium',
            bitrate="1000k",
            audio_codec='aac',
            threads=12
        )

        # Clean up image and audio files
        for i in range(len(video_inputs)):
            line_id = f"{i:03}"
            image_file = f"latex_image_{line_id}.png"
            audio_file = f"{i:03}audio.mp3"
            if os.path.exists(image_file):
                os.remove(image_file)
            if os.path.exists(audio_file):
                os.remove(audio_file)

        print(f"Total time taken: {time.time() - start_time} seconds.")
        return output_video
