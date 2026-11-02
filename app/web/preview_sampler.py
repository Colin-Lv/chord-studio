"""Local piano preview renderer, extracted unchanged from our dataset sampler.

The sampler and rendering operations are bundled with the application. No research
folder, training dataset, or external Python import is needed for MIDI audition.
"""
import numpy as np
import pretty_midi
from . import sf2_sampler as sf2

SR = 48000
TAIL = 0.08
FADE = 0.06
MAX_RELEASE = 0.60

def make_sampler_class(sf2):
    class DatasetSoundFont(sf2.SoundFont):
        """Sampler with cached preset zones and a bounded release tail."""

        def __init__(self, path):
            super().__init__(path)
            self._zones: dict[tuple[int, int], list] = {}

        def preset_zones(self, bank: int, program: int):
            key = (bank, program)
            if key not in self._zones:
                zones = super().preset_zones(bank, program)
                for z in zones:
                    z.release = min(z.release, MAX_RELEASE)
                self._zones[key] = zones
            return self._zones[key]

    return DatasetSoundFont

def render_pm(pm: pretty_midi.PrettyMIDI, sfx, sr: int = SR, gain: float = 0.65
              ) -> np.ndarray:
    total = max(0.5, pm.get_end_time()) + 0.8
    n = int(total * sr)
    mix = np.zeros(n, dtype=np.float32)
    for inst in pm.instruments:
        program = int(inst.program)
        for note in inst.notes:
            dur = max(0.03, note.end - note.start)
            audio, _pan = sfx.render(program, note.pitch, note.velocity,
                                     dur + TAIL, sr=sr, bank=0, gain=gain)
            if audio.size < 8:
                continue
            fade = min(int(FADE * sr), audio.size)
            audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            start = int(round(note.start * sr))
            length = min(audio.size, n - start)
            if length <= 0:
                continue
            mix[start:start + length] += audio[:length]
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0:
        mix *= 0.89 / peak
    return mix
