import numpy as np
from moviepy import *
# from moviepy.video.tools.segmenting import findObjects

class TextAnimator:
    def __init__(self, text, font="Amiri-Bold", fontsize=100, color='white', kerning=5, screensize=(720, 460)):
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
        return lambda t: screenpos + 400 * d(t) * self.rotMatrix(0.5 * d(t) * a).dot(v)

    def cascade(self, screenpos, i, nletters):
        v = np.array([0, -1])
        d = lambda t: 1 if t < 0 else abs(np.sinc(t) / (1 + t ** 4))
        return lambda t: screenpos + v * 400 * d(t - 0.15 * i)

    def arrive(self, screenpos, i, nletters):
        v = np.array([-1, 0])
        d = lambda t: max(0, 3 - 3 * t)
        return lambda t: screenpos - 400 * v * d(t - 0.2 * i)

    def vortexout(self, screenpos, i, nletters):
        d = lambda t: max(0, t)  # damping
        a = i * np.pi / nletters  # angle of the movement
        v = self.rotMatrix(a).dot([-1, 0])
        if i % 2:
            v[1] = -v[1]
        return lambda t: screenpos + 400 * d(t - 0.1 * i) * self.rotMatrix(-0.2 * d(t) * a).dot(v)

    def animate_text(self, effect='cascade', duration=5):
        # Create the TextClip and center it
        txtClip = TextClip(
            self.text,
            color=self.color,
            font=self.font,
            kerning=self.kerning,
            fontsize=self.fontsize
        )
        cvc = CompositeVideoClip([txtClip.set_pos('center')], size=self.screensize)

        # Use findObjects to locate and separate each letter
        letters = findObjects(cvc)

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
        def moveLetters(letters, funcpos):
            return [
                letter.set_pos(funcpos(letter.screenpos, i, len(letters)))
                for i, letter in enumerate(letters)
            ]

        # Create the animated clip
        animated_clip = CompositeVideoClip(
            moveLetters(letters, funcpos),
            size=self.screensize
        ).subclip(0, duration)

        return animated_clip
