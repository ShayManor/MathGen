from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.compositing.transitions import crossfadein
from moviepy.video.io.VideoFileClip import VideoFileClip


def concatenate_videos_with_transitions(file_list, output_file='output_video.mp4', transition_duration=1):
    try:
        clips = [VideoFileClip(file) for file in file_list]

        # Add crossfade transition between each clip
        final_clip = clips[0]
        for clip in clips[1:]:
            final_clip = concatenate_videoclips([final_clip, transition_duration], method='compose', transition=crossfadein())

        final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

        # Close all clips
        for clip in clips:
            clip.close()
        final_clip.close()

        print(f"Successfully concatenated videos into {output_file} with transitions")

    except Exception as e:
        print(f"An error occurred: {e}")