import time

from matplotlib import pyplot as plt
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips

from Movie_Assembler.create_movie import script_to_audio
from Movie_Creator.video_input import video_input
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.audio.AudioClip

from typing_extensions import final


class section:
    def __init__(self):
        self.audio_path = None
        self.image_path = None
        self.start_time = 0
        self.duration = 0


class create_movie2:
    def __init__(self, api_key):
        self.api_key = api_key

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

    def create_video_from_inputs(self, video_inputs: list[video_input], output_video='final_movie.mp4'):
        lines = []
        sections = []
        times = []
        audio_clips = []
        t3 = time.time()
        for index, v_inp in enumerate(video_inputs):
            sect = section()

            script_to_audio(self.api_key).convert(v_inp.script, index)
            line = v_inp.on_screen
            lines.append(line)

            sect.audio_path = f"{index:03}audio.mp3"

            clip = AudioFileClip(sect.audio_path)
            times.append(clip.duration + sum(times))
            sect.duration = clip.duration
            sect.start_time = sum(times[:index])  # Corrected slicing
            times.append(sect.duration)  # Populate the times list

            sections.append(sect)
        print(f'Time to \"record\" audios: {time.time()-t3}')
        visible_lines: list[str] = []

        for index, line in enumerate(lines):

            if index > 0 and line == lines[index - 1]:
                sections[index - 1].duration += sections[index].duration
                times.remove(times[index])
                continue

            visible_lines.append(line)

            if len(visible_lines) > 3:
                visible_lines.pop()

            cumulative_text = r' \\ '.join(visible_lines)
            cumulative_text = r'\begin{align*}' + cumulative_text + r'\end{align*}'

            # Render LaTeX image for the current set of visible lines
            image_file = f"{index:03}latex_image.png"
            self.render_latex_to_image(cumulative_text, image_file)
            sections[index].image_path = image_file

        final_clips = []
        final_duration = 0
        for index, sect in enumerate(sections):
            audio = AudioFileClip(sect.audio_path)
            final_duration += audio.duration
            image = ImageClip(sect.image_path)
            image.set_audio(audio)
            image.set_duration(audio.duration)
            final_clips.append(image)
            print(image)
            print(audio)

        print(final_clips[0].duration)
        final_video: concatenate_videoclips = concatenate_videoclips(final_clips, method="compose")
        # final_video.set_fps(4)
        final_video.write_videofile(output_video, fps=24)
        return final_video


# v = video_input()
# v.set_script('abc')
# v.set_on_screen('def')
# create_movie2(
#     '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').create_video_from_inputs([v])
