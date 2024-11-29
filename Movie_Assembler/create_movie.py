import re
import tempfile
import time
from uuid import uuid3
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import os
from moviepy.config import change_settings

from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_audioclips
)
from moviepy.video.VideoClip import TextClip, ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
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
        change_settings({"IMAGEMAGICK_BINARY": "magick"})

    def make_silence(self, duration, fps=44100, n_channels=2):
        total_samples = int(duration * fps)
        array = np.zeros((total_samples, n_channels))
        return AudioArrayClip(array, fps=fps)

    def create_animated_text_clip(self, latex_str, duration):
        text_clip = TextClip(
            latex_str,
            fontsize=50,
            color='white',
            font='Arial',
            method='latex',
        ).set_duration(duration)

        animated_clip = text_clip.crossfadein(1)  # 1 second fade-in
        return animated_clip

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

    import tempfile
    import os

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
        fig, ax = plt.subplots(figsize=(12, 6))
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
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=1, dpi=200)
            print(f"LaTeX rendered successfully to {output_image_path}")
        except Exception as e:
            print(f'Error with LaTeX rendering: {e}')
            raise  # Re-raise the exception to handle it upstream
        finally:
            plt.close(fig)

        # Check if the image was created
        if not os.path.exists(output_image_path):
            print(f"Failed to create image: {output_image_path}")
            return None
        else:
            print(f"Image {output_image_path} exists.")
            return output_image_path

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

        # Function to sanitize individual lines
        def sanitize_line(line):
            line = line.strip()
            if line.startswith(r'\begin{align*}') and line.endswith(r'\end{align*}'):
                # Remove the outer align* environment
                line = line[len(r'\begin{align*}'): -len(r'\end{align*}')]
            elif line.startswith('$$') and line.endswith('$$'):
                # Remove $$ delimiters
                line = line[2:-2]
            return line

        # Collect start and end times for each line
        for i, video_input in enumerate(video_inputs):
            current_line = self.escape_latex(video_input.on_screen)
            current_line = sanitize_line(current_line)  # Sanitize the line
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
        current_visible_lines = []  # Changed from set to list
        prev_time = 0.0

        for event in events:
            event_time, event_type, line_text = event
            if event_time > prev_time:
                # There is an interval from prev_time to event_time
                if current_visible_lines:
                    cumulative_text = r'\begin{align*}' + r' \\ '.join(current_visible_lines) + r'\end{align*}'
                else:
                    cumulative_text = ''  # Handle the case where no lines are visible
                intervals.append((prev_time, event_time, cumulative_text))
                prev_time = event_time
            # Update current_visible_lines
            if event_type == 'start':
                # Insert at the beginning to make it the top line
                current_visible_lines.insert(0, line_text)
            elif event_type == 'end':
                # Remove the line from the list
                if line_text in current_visible_lines:
                    current_visible_lines.remove(line_text)

        # Create ImageClips for each interval
        image_clips = []

        for idx, (start_time_interval, end_time_interval, cumulative_text) in enumerate(intervals):
            duration = end_time_interval - start_time_interval
            if duration <= 0:
                continue  # skip zero or negative duration intervals
            image_file = f"latex_image_interval_{idx}.png"
            if cumulative_text:
                image_path = self.render_latex_to_image(cumulative_text, image_file)
                if image_path:
                    image_clip = ImageClip(image_path)
                    image_clip = image_clip.set_duration(duration)
                    image_clips.append(image_clip)
                else:
                    print(f"Skipping ImageClip creation for interval {idx} due to rendering failure.")
            else:
                # Create a blank image if no text is visible
                blank_image_path = os.path.join(os.getcwd(), 'images', "blank_image.png")
                if not os.path.exists(blank_image_path):
                    # Create and save a blank image
                    plt.figure(figsize=(6, 3))
                    plt.axis('off')
                    plt.savefig(blank_image_path, bbox_inches='tight', pad_inches=0.1, dpi=50)
                    plt.close()
                image_clip = ImageClip(blank_image_path).set_duration(duration)
                image_clips.append(image_clip)

        # Optionally, add fade-in and fade-out effects to each image clip
        for i in range(len(image_clips)):
            image_clips[i] = image_clips[i].fadein(1).fadeout(1)

        # Concatenate ImageClips
        video_clip = concatenate_videoclips(image_clips, method='compose')
        video_clip = video_clip.set_audio(final_audio)

        # Write the video file
        try:
            video_clip.write_videofile(
                output_video,
                fps=24,  # Increased fps for smoother transitions
                codec='libx264',
                preset='medium',  # Changed preset for better encoding
                audio_codec='aac',
                threads=4,
                logger='bar'  # Enable logging to monitor progress
            )
            print(f"Video successfully created: {output_video}")
        except Exception as e:
            print(f"Error writing video file: {e}")

        # Clean up image and audio files
        for idx in range(len(intervals)):
            image_file = f"latex_image_interval_{idx}.png"
            image_path = os.path.join(os.getcwd(), 'images', image_file)
            if os.path.exists(image_path):
                os.remove(image_path)
        for i in range(len(video_inputs)):
            audio_file = f"{i:03}audio.mp3"
            audio_path = os.path.join(os.getcwd(), audio_file)
            if os.path.exists(audio_path):
                os.remove(audio_path)
        blank_image_path = os.path.join(os.getcwd(), 'images', "blank_image.png")
        if os.path.exists(blank_image_path):
            os.remove(blank_image_path)

        print(f"Total time taken: {time.time() - start_time} seconds.")
        return output_video
