import numpy as np
from moviepy import *

class TextAnimator:
    def __init__(self, text, font="Arial", fontsize=100, color='white', kerning=5, screensize=(720, 460)):
        self.text = text
        self.font = font
        self.fontsize = fontsize
        self.color = color
        self.kerning = kerning
        self.screensize = screensize

    def rotMatrix(self, a):
        return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def vortex(self, screenpos, i, nletters):
        d = lambda t: 1.0 / (0.3 + t ** 8)  # damping
        a = i * np.pi / nletters  # angle of the movement
        v = self.rotMatrix(a).dot([-1, 0])
        if i % 2:
            v[1] = -v[1]
        return lambda t: (
            screenpos[0] + 400 * d(t) * self.rotMatrix(0.5 * d(t) * a).dot(v)[0],
            screenpos[1] + 400 * d(t) * self.rotMatrix(0.5 * d(t) * a).dot(v)[1]
        )

    def cascade(self, screenpos, i, nletters):
        v = np.array([0, -1])
        d = lambda t: 1 if t < 0 else abs(np.sinc(t) / (1 + t ** 4))
        return lambda t: (
            screenpos[0],
            screenpos[1] + v[1] * 400 * d(t - 0.15 * i)
        )

    def arrive(self, screenpos, i, nletters):
        v = np.array([-1, 0])
        d = lambda t: max(0, 3 - 3 * t)
        return lambda t: (
            screenpos[0] - 400 * v[0] * d(t - 0.2 * i),
            screenpos[1] - 400 * v[1] * d(t - 0.2 * i)
        )

    def vortexout(self, screenpos, i, nletters):
        d = lambda t: max(0, t)  # damping
        a = i * np.pi / nletters  # angle of the movement
        v = self.rotMatrix(a).dot([-1, 0])
        if i % 2:
            v[1] = -v[1]
        return lambda t: (
            screenpos[0] + 400 * d(t - 0.1 * i) * self.rotMatrix(-0.2 * d(t) * a).dot(v)[0],
            screenpos[1] + 400 * d(t - 0.1 * i) * self.rotMatrix(-0.2 * d(t) * a).dot(v)[1]
        )

    def create_letter_clips(self):
        letters = []
        total_width = 0
        # Measure the width of each character
        for char in self.text:
            letter = TextClip(
                font='Arial',
                text=char,
                # font=self.font,
                font_size=self.fontsize,
                color=self.color,
                # kerning=self.kerning,
                transparent=True
            )
            letters.append(letter)
            total_width += letter.w

        # Calculate starting positions to center the text
        positions = []
        current_x = (self.screensize[0] - total_width) / 2
        for letter in letters:
            positions.append((current_x, (self.screensize[1] - letter.h) / 2))
            current_x += letter.w

        # Assign positions to each letter
        for letter, pos in zip(letters, positions):
            letter.screenpos = pos

        return letters

    def animate_text(self, effect='cascade', duration=5):
        # Create letter clips
        letters = self.create_letter_clips()

        # Map the effect names to the corresponding methods
        effects_dict = {
            'vortex': self.vortex,
            'cascade': self.cascade,
            'arrive': self.arrive,
            'vortexout': self.vortexout
        }

        if effect not in effects_dict:
            raise ValueError(f"Effect '{effect}' not recognized. Available effects: {list(effects_dict.keys())}")

        funcpos = effects_dict[effect]

        # Function to animate letters
        def move_letters(letters: list[TextClip], funcpos):
            for i, letter in enumerate(letters):
                letter.pos = funcpos(letter.pos, i, len(letters))
                letter.start = 0
                letter.duration = duration
            return letters

        # Create the animated clip
        animated_letters = move_letters(letters, funcpos)
        animated_clip = CompositeVideoClip(animated_letters, size=self.screensize)
        animated_clip.duration = duration

        return animated_clip
