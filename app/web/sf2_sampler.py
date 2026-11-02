"""Minimal SoundFont 2 (.sf2) reader and sampler, numpy only.

Supports what GM rendering needs: preset -> instrument -> zone -> sample, key/velocity
ranges, root key and tuning, loop points, volume envelope (attack/decay/sustain/release)
and attenuation. Stereo samples are averaged to mono; 24-bit samples are not handled
(FluidR3_GM is 16-bit, which is what this was written for).

    sf = SoundFont("FluidR3_GM.sf2")
    audio = sf.render(0, note=60, velocity=90, duration=1.5, sr=44100)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GEN_KEY_RANGE, GEN_VEL_RANGE = 43, 44
GEN_START_OFF, GEN_END_OFF = 0, 1
GEN_STARTLOOP_OFF, GEN_ENDLOOP_OFF = 2, 3
GEN_START_COARSE, GEN_END_COARSE = 4, 12
GEN_STARTLOOP_COARSE, GEN_ENDLOOP_COARSE = 45, 50
GEN_INITIAL_ATTENUATION = 48
GEN_PAN = 17
GEN_INSTRUMENT = 41
GEN_SAMPLE_ID = 53
GEN_SAMPLE_MODES = 54
GEN_SCALE_TUNING = 56
GEN_OVERRIDING_ROOT = 58
GEN_COARSE_TUNE, GEN_FINE_TUNE = 51, 52
GEN_ATTACK, GEN_HOLD, GEN_DECAY, GEN_SUSTAIN, GEN_RELEASE = 34, 35, 36, 37, 38
GEN_KEYNUM_TO_DECAY = 40
SIGNED_GENS = {34, 35, 36, 38, 51, 52, 17}
GEN_FILTER_FC, GEN_FILTER_Q = 8, 9


def _tc_to_seconds(tc: int) -> float:
    return 2.0 ** (tc / 1200.0)


def _cb_to_gain(cb: int) -> float:
    return 10.0 ** (-cb / 200.0)


def _signed(value: int) -> int:
    return value - 65536 if value > 32767 else value


@dataclass
class Zone:
    key_low: int = 0
    key_high: int = 127
    vel_low: int = 0
    vel_high: int = 127
    root: int | None = None
    tuning: float = 0.0
    scale: int = 100
    loop: bool = False
    sample_id: int | None = None
    instrument: int | None = None
    attenuation: float = 0.0
    pan: float = 0.0
    attack: float = 0.001
    hold: float = 0.0
    decay: float = 0.5
    sustain: float = 1.0
    release: float = 0.1
    gen: dict = field(default_factory=dict)


class SoundFont:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        self._read_riff()
        self._read_hydra()

    # ---------------- RIFF ----------------
    def _read_riff(self):
        raw = self.raw
        if raw[:4] != b"RIFF" or raw[8:12] != b"sfbk":
            raise ValueError("not a SoundFont 2 file")
        pos, self.smpl, self.info, self.pdta = 12, b"", {}, {}
        while pos + 8 <= len(raw):
            cid = raw[pos:pos + 4]
            size = struct.unpack_from("<I", raw, pos + 4)[0]
            body = raw[pos + 8:pos + 8 + size]
            if cid == b"LIST":
                kind = body[:4]
                sub, spos = {}, 4
                while spos + 8 <= len(body):
                    sid = body[spos:spos + 4]
                    ssize = struct.unpack_from("<I", body, spos + 4)[0]
                    sub[sid.decode("latin-1")] = body[spos + 8:spos + 8 + ssize]
                    spos += 8 + ssize + (ssize % 2)
                if kind == b"sdta":
                    self.smpl = sub.get("smpl", b"")
                elif kind == b"INFO":
                    self.info = {k: v.split(b"\x00")[0].decode("latin-1", "replace") for k, v in sub.items()}
                elif kind == b"pdta":
                    self.pdta = sub
            pos += 8 + size + (size % 2)

    def _records(self, name: str, fmt: str):
        data = self.pdta.get(name, b"")
        size = struct.calcsize(fmt)
        return [struct.unpack_from(fmt, data, i) for i in range(0, len(data) - size + 1, size)]

    def _read_hydra(self):
        self.phdr = self._records("phdr", "<20sHHHIII")
        self.pbag = self._records("pbag", "<HH")
        self.pgen = self._records("pgen", "<HH")
        self.inst = self._records("inst", "<20sH")
        self.ibag = self._records("ibag", "<HH")
        self.igen = self._records("igen", "<HH")
        self.shdr = self._records("shdr", "<20sIIIIIBbHH")
        self.presets = [(p[1], p[2], p[0].split(b"\x00")[0].decode("latin-1", "replace")) for p in self.phdr]

    # ---------------- hydra helpers ----------------
    @staticmethod
    def _gen_map(rows, start: int, end: int) -> dict:
        out = {}
        for op, amount in rows[start:end]:
            out[op] = amount
        return out

    def _instrument_zones(self, instrument_index: int):
        start = self.inst[instrument_index][1]
        end = self.inst[instrument_index + 1][1]
        zones = []
        for i in range(start, end):
            z = Zone()
            z.gen = self._gen_map(self.igen, self.ibag[i][0], self.ibag[i + 1][0])
            self._apply_gens(z)
            if z.sample_id is not None:
                zones.append(z)
        return zones

    def _apply_gens(self, z: Zone):
        g = z.gen
        if GEN_KEY_RANGE in g:
            z.key_low, z.key_high = g[GEN_KEY_RANGE] & 0xFF, (g[GEN_KEY_RANGE] >> 8) & 0xFF
        if GEN_VEL_RANGE in g:
            z.vel_low, z.vel_high = g[GEN_VEL_RANGE] & 0xFF, (g[GEN_VEL_RANGE] >> 8) & 0xFF
        if GEN_OVERRIDING_ROOT in g:
            z.root = g[GEN_OVERRIDING_ROOT]
        z.tuning = (g.get(GEN_COARSE_TUNE, 0) * 100.0) + (g.get(GEN_FINE_TUNE, 0) / 100.0)
        z.scale = g.get(GEN_SCALE_TUNING, 100)
        z.loop = g.get(GEN_SAMPLE_MODES, 0) in (1, 3)
        z.sample_id = g.get(GEN_SAMPLE_ID)
        z.attenuation = _cb_to_gain(g.get(GEN_INITIAL_ATTENUATION, 0))
        z.pan = float(np.clip(g.get(GEN_PAN, 0) / 500.0, -1.0, 1.0))
        z.attack = _tc_to_seconds(_signed(g.get(GEN_ATTACK, -12000)))
        z.hold = _tc_to_seconds(_signed(g.get(GEN_HOLD, -12000)))
        z.decay = _tc_to_seconds(_signed(g.get(GEN_DECAY, -12000)))
        z.sustain = _cb_to_gain(g.get(GEN_SUSTAIN, 0))
        z.release = _tc_to_seconds(_signed(g.get(GEN_RELEASE, -12000)))

    def preset_zones(self, bank: int, program: int):
        for index, p in enumerate(self.phdr):
            name, preset, pbank, bag_index = p[0], p[1], p[2], p[3]
            if preset != program or pbank != bank:
                continue
            zones = []
            nxt = self.phdr[index + 1][3] if index + 1 < len(self.phdr) else len(self.pbag)
            for i in range(bag_index, nxt):
                g = self._gen_map(self.pgen, self.pbag[i][0], self.pbag[i + 1][0])
                inst_index = g.get(GEN_INSTRUMENT)
                if inst_index is None:
                    continue
                for zone in self._instrument_zones(inst_index):
                    if GEN_KEY_RANGE in g:
                        lo, hi = g[GEN_KEY_RANGE] & 0xFF, (g[GEN_KEY_RANGE] >> 8) & 0xFF
                        zone.key_low, zone.key_high = max(zone.key_low, lo), min(zone.key_high, hi)
                    if GEN_VEL_RANGE in g:
                        lo, hi = g[GEN_VEL_RANGE] & 0xFF, (g[GEN_VEL_RANGE] >> 8) & 0xFF
                        zone.vel_low, zone.vel_high = max(zone.vel_low, lo), min(zone.vel_high, hi)
                    zones.append(zone)
            return zones
        return []

    def sample(self, index: int):
        name, start, end, loop_start, loop_end, rate, root, correction, link, stype = self.shdr[index]
        # shdr offsets are sample frames; the smpl chunk stores 16-bit words.
        pcm = np.frombuffer(self.smpl[start * 2:end * 2], dtype="<i2").astype(np.float32) / 32768.0
        stereo = bool(stype & 2)
        if stereo:                          # keep the left channel
            pcm = pcm[0::2]
            loop_start //= 2
            loop_end //= 2
        return dict(name=name.split(b"\x00")[0].decode("latin-1", "replace"), pcm=pcm,
                    loop_start=max(0, loop_start - start),
                    loop_end=max(0, loop_end - start),
                    rate=rate, root=root, correction=correction / 100.0,
                    stereo=stereo)

    # ---------------- playback ----------------
    def render(self, program: int, note: int, velocity: int, duration: float,
               sr: int = 44100, bank: int = 0, gain: float = 1.0, release_extra: float = 0.25):
        all_zones = self.preset_zones(bank, program)
        zones = [z for z in all_zones if z.key_low <= note <= z.key_high
                 and z.vel_low <= velocity <= z.vel_high]
        if not zones:
            zones = [z for z in all_zones if z.key_low <= note <= z.key_high] or all_zones
        if not zones:
            return np.zeros(int((duration + release_extra) * sr), dtype=np.float32), 0.0
        # GM presets stack many velocity/key layers; take the one whose root is
        # closest to the note, then the velocity layer closest to the request.
        zone = min(zones, key=lambda z: (
            abs((z.root if z.root is not None else 60) - note),
            abs((z.vel_low + z.vel_high) / 2 - velocity)))
        info = self.sample(zone.sample_id)
        pcm = info["pcm"]
        if pcm.size < 8:
            return np.zeros(int((duration + release_extra) * sr), dtype=np.float32), 0.0
        root = zone.root if zone.root is not None else info["root"]
        ratio = 2.0 ** ((note - root) / 12.0 + info["correction"] / 12.0) * (info["rate"] / sr)
        length = int((duration + zone.release) * sr) + 1
        idx = np.arange(length) * ratio
        loop_start, loop_end = info["loop_start"], info["loop_end"]
        if zone.loop and loop_end > loop_start + 8:
            span = loop_end - loop_start
            idx = np.where(idx < loop_end, idx, loop_start + (idx - loop_start) % span)
        idx = np.clip(idx, 0, len(pcm) - 2)
        i0 = idx.astype(np.int64)
        frac = idx - i0
        audio = pcm[i0] * (1 - frac) + pcm[i0 + 1] * frac
        t = np.arange(len(audio)) / sr
        env = self._envelope(t, zone, duration)
        audio = audio * env * zone.attenuation * gain * (velocity / 127.0) ** 0.8
        return audio.astype(np.float32), zone.pan

    @staticmethod
    def _envelope(t: np.ndarray, zone: Zone, duration: float):
        attack = max(zone.attack, 1e-4)
        hold = max(zone.hold, 0.0)
        decay = max(zone.decay, 1e-3)
        sustain = float(np.clip(zone.sustain, 0.0, 1.0))
        env = np.ones_like(t)
        env = np.where(t < attack, t / attack, env)
        after_hold = t >= attack + hold
        shape = sustain + (1.0 - sustain) * np.exp(-(t - attack - hold) / decay)
        env = np.where(after_hold, shape, env)
        if duration < t[-1]:
            rel = np.clip((t - duration) / max(zone.release, 1e-3), 0, 1)
            env = env * (1 - rel)
        return np.clip(env, 0, 1)
