import re
import time
from uuid import uuid3
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os

from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_audioclips
)
from moviepy.video.compositing.concatenate import concatenate_videoclips
from openai import OpenAI


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

        # Debug: Print the final LaTeX string to be rendered
        print(f"Final LaTeX string to render: {latex_str}")

        # Render the LaTeX string
        ax.text(
            0.5, 0.5, latex_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
        try:
            plt.savefig(output_image, bbox_inches='tight', pad_inches=1, dpi=200)
        except Exception as e:
            print(f'Error with LaTeX shown on screen: {e}')
            raise  # Re-raise the exception to handle it upstream
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
        current_time = 0.0
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

        # Prepare video clips
        final_duration = final_audio.duration

        # Initialize variables
        visible_lines = []
        line_start_times = {}
        line_end_times = {}
        events = []

        # Collect start and end times for each line
        for i, video_input in enumerate(video_inputs):
            current_line = self.escape_latex(video_input.on_screen)
            start_time_line = start_times[i]
            line_start_times[current_line] = start_time_line
            events.append((start_time_line, 'start', current_line))

            # Check if the line is already visible
            if current_line in visible_lines:
                print(f"Line already visible, skipping duplicate: {current_line}")
                continue  # Skip adding duplicate lines

            visible_lines.append(current_line)

            # If more than 3 lines are visible, remove the oldest one
            if len(visible_lines) > 3:
                removed_line = visible_lines.pop(0)
                end_time_line = start_time_line  # Current time
                line_end_times[removed_line] = end_time_line
                events.append((end_time_line, 'end', removed_line))

        # Set end times for remaining lines
        for remaining_line in visible_lines:
            line_end_times[remaining_line] = final_duration
            events.append((final_duration, 'end', remaining_line))

        # Sort events by time
        events.sort()

        # Process events to build intervals
        intervals = []
        current_visible_lines = set()
        prev_time = 0.0

        for event in events:
            event_time, event_type, line_text = event
            if event_time > prev_time:
                # There is an interval from prev_time to event_time
                cumulative_text = r' \\ '.join(current_visible_lines)
                if cumulative_text:
                    cumulative_text = r'\begin{align*}' + cumulative_text + r'\end{align*}'
                else:
                    cumulative_text = ''  # Handle the case where no lines are visible
                intervals.append((prev_time, event_time, cumulative_text))
                prev_time = event_time
            # Update current_visible_lines
            if event_type == 'start':
                current_visible_lines.add(line_text)
            elif event_type == 'end':
                current_visible_lines.remove(line_text)

        # Create ImageClips for each interval
        image_clips = []

        for idx, (start_time_interval, end_time_interval, cumulative_text) in enumerate(intervals):
            duration = end_time_interval - start_time_interval
            if duration <= 0:
                continue  # skip zero or negative duration intervals
            image_file = f"latex_image_interval_{idx}.png"
            if cumulative_text:
                self.render_latex_to_image(cumulative_text, image_file)
            else:
                # Create a blank image if no text is visible
                image_file = "blank_image.png"
                if not os.path.exists(image_file):
                    plt.figure(figsize=(6, 3))
                    plt.axis('off')
                    plt.savefig(image_file, bbox_inches='tight', pad_inches=0.1, dpi=50)
                    plt.close()
            # Create ImageClip
            try:
                image_clip = ImageClip(image_file)
                image_clip = image_clip.set_duration(duration)
                image_clips.append(image_clip)
            except:
                print("Error rendering latex")

        # Concatenate ImageClips
        video_clip = concatenate_videoclips(image_clips, method='compose')
        video_clip = video_clip.set_audio(final_audio)

        # Write the video file
        video_clip.write_videofile(
            output_video,
            fps=1,  # Lower fps since images are static
            codec='libx264',
            preset='ultrafast',  # Use ultrafast preset to speed up encoding
            audio_codec='aac',
            threads=4,
            logger=None  # Disable verbose logging
        )

        # Clean up image and audio files
        for idx in range(len(intervals)):
            image_file = f"latex_image_interval_{idx}.png"
            if os.path.exists(image_file):
                os.remove(image_file)
        for i in range(len(video_inputs)):
            audio_file = f"{i:03}audio.mp3"
            if os.path.exists(audio_file):
                os.remove(audio_file)
        if os.path.exists("blank_image.png"):
            os.remove("blank_image.png")

        print(f"Total time taken: {time.time() - start_time} seconds.")
        return output_video
