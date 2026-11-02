"""Offline FluidSynth float renderer; one absolute clock and no per-track normalization."""
import ctypes as C
import os
from pathlib import Path
import numpy as np

from app.runtime_paths import soundfont_path, fluidsynth_path
SOUNDFONT = soundfont_path()
DLL = fluidsynth_path()


class Synth:
    def __init__(self, rate=22050):
        self.directory = os.add_dll_directory(str(DLL.parent)) if os.name=='nt' else None
        self.lib = C.CDLL(str(DLL))
        self.bind('new_fluid_settings', C.c_void_p)
        self.bind('fluid_settings_setnum', C.c_int, C.c_void_p,C.c_char_p,C.c_double)
        self.bind('fluid_settings_setint', C.c_int, C.c_void_p,C.c_char_p,C.c_int)
        self.bind('new_fluid_synth', C.c_void_p,C.c_void_p)
        self.bind('fluid_synth_sfload', C.c_int,C.c_void_p,C.c_char_p,C.c_int)
        self.bind('fluid_synth_program_select', C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int)
        for name,argc in [('noteon',3),('noteoff',2),('cc',3),('pitch_bend',2)]:
            self.bind('fluid_synth_'+name,C.c_int,C.c_void_p,*([C.c_int]*argc))
        self.bind('fluid_synth_write_float',C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int)
        self.bind('delete_fluid_synth',None,C.c_void_p);self.bind('delete_fluid_settings',None,C.c_void_p)
        self.settings=self.lib.new_fluid_settings()
        self.lib.fluid_settings_setnum(self.settings,b'synth.sample-rate',rate)
        self.lib.fluid_settings_setnum(self.settings,b'synth.gain',.2)
        self.lib.fluid_settings_setint(self.settings,b'synth.reverb.active',0)
        self.lib.fluid_settings_setint(self.settings,b'synth.chorus.active',0)
        self.synth=self.lib.new_fluid_synth(self.settings)
        self.sf=self.lib.fluid_synth_sfload(self.synth,os.fsencode(SOUNDFONT),0)
        if self.sf<0:self.close();raise RuntimeError('无法加载试听音源')

    def bind(self,name,restype,*args):
        f=getattr(self.lib,name);f.restype=restype;f.argtypes=list(args)

    def close(self):
        if getattr(self,'synth',None):self.lib.delete_fluid_synth(self.synth);self.synth=None
        if getattr(self,'settings',None):self.lib.delete_fluid_settings(self.settings);self.settings=None
        if self.directory:self.directory.close();self.directory=None

    def samples(self,count):
        out=np.zeros((count,2),dtype=np.float32)
        self.lib.fluid_synth_write_float(self.synth,count,out.ctypes.data,0,2,out.ctypes.data,1,2)
        return out


def render(path,duration,rate=22050):
    import pretty_midi
    midi=pretty_midi.PrettyMIDI(str(path))
    synth=Synth(rate)
    length=round(duration*rate);out=np.zeros((length,2),dtype=np.float32)
    try:
        events=[];melodic=[c for c in range(16) if c!=9]
        pitched=0
        for inst in midi.instruments:
            if not inst.is_drum and pitched>=len(melodic):raise ValueError('单个试听文件超过 15 个有音高声部，请分轨渲染')
            ch=9 if inst.is_drum else melodic[pitched]
            if not inst.is_drum:pitched+=1
            if synth.lib.fluid_synth_program_select(synth.synth,ch,synth.sf,128 if inst.is_drum else 0,inst.program)<0:
                raise ValueError('音源中没有指定乐器')
            for n in inst.notes:
                events.extend([(round(n.start*rate),2,'noteon',(ch,n.pitch,n.velocity)),(round(n.end*rate),0,'noteoff',(ch,n.pitch))])
            for c in inst.control_changes:events.append((round(c.time*rate),1,'cc',(ch,c.number,c.value)))
            for b in inst.pitch_bends:events.append((round(b.time*rate),1,'pitch_bend',(ch,b.pitch+8192)))
        cursor=0
        for sample,_,kind,args in sorted(events,key=lambda e:(e[0],e[1])):
            sample=min(length,max(0,sample))
            if sample>cursor:out[cursor:sample]=synth.samples(sample-cursor);cursor=sample
            getattr(synth.lib,'fluid_synth_'+kind)(synth.synth,*args)
        if cursor<length:out[cursor:]=synth.samples(length-cursor)
        return out
    finally:synth.close()


def reference_audio(path,duration,rate=22050):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    y,sr=sf.read(path,dtype='float32',always_2d=True)
    if sr!=rate:
        g=gcd(sr,rate);y=resample_poly(y,rate//g,sr//g,axis=0).astype(np.float32)
    if y.shape[1]==1:y=np.repeat(y,2,axis=1)
    n=round(duration*rate)
    return np.pad(y[:n,:2],((0,max(0,n-len(y))),(0,0)))


def fit_level(rendered,reference):
    # Only a bounded global playback gain, never a claim of timbre recovery.
    ref=float(np.sqrt(np.mean(reference.astype(np.float64)**2)))
    rms=float(np.sqrt(np.mean(rendered.astype(np.float64)**2)))
    gain=float(np.clip(ref/max(rms,1e-9),.125,8.)) if rms>1e-9 else 1.
    return gain,dict(reference_rms=ref,render_rms=rms,gain=gain,fit='bounded global RMS; no note editing')
