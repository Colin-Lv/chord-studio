"use strict";

// One rendering clock for every stem. Gain changes never seek or pause a source;
// visual animation frames are deliberately not part of audio synchronization.
class StemTransport {
  constructor(contextFactory = () => new AudioContext({ latencyHint: "interactive" })) {
    this.contextFactory = contextFactory;this.context = null;this.job = null;
    this.sources = [];this.live = new Set();this.clock = null;this.generation = 0;
    this.duration = 0;this.offset = 0;this.anchor = null;this.playing = false;
    this.range = null;this.levels = {};this.onEnded = null;
  }
  async unlock() {
    if (!this.context) this.context = this.contextFactory();
    if (!this.context.startRendering && this.context.state !== "running") await this.context.resume();
  }
  reset(duration, onEnded) {
    this.dispose();this.duration = duration;this.onEnded = onEnded;
  }
  dispose() {
    this.pause();
    const job = this.job;this.job = null;
    if (job) {
      job.controller.abort();
      for (const record of job.records.values()) record.gain.disconnect();
      job.records.clear();
    }
    this.offset = 0;this.range = null;this.levels = {};this.onEnded = null;
  }
  load(tracks) {
    if (this.job) return this.job.promise;
    if (!this.context) throw new Error("音频设备尚未准备，请重新点击播放。");
    const job = { controller: new AbortController(), records: new Map(), promise: null };
    this.job = job;let cursor = 0;
    const active = () => {
      if (job.controller.signal.aborted || this.job !== job) throw new DOMException("Cancelled", "AbortError");
    };
    const worker = async () => {
      while (cursor < tracks.length) {
        active();const track = tracks[cursor++];
        const response = await fetch(track.src, { signal: job.controller.signal });
        if (!response.ok) throw new Error("读取试听音频失败（" + response.status + "），请重试。");
        const bytes = await response.arrayBuffer();active();
        let buffer;
        try { buffer = await this.context.decodeAudioData(bytes); }
        catch (error) { active();throw new Error("试听音频无法解码，请重新加载。"); }
        active();
        // Short encoder tails must not give individual stems a shorter loop.
        const frames = Math.ceil(this.duration * buffer.sampleRate);
        if (buffer.length < frames) {
          const padded = this.context.createBuffer(buffer.numberOfChannels, frames, buffer.sampleRate);
          for (let c = 0; c < buffer.numberOfChannels; c++) padded.copyToChannel(buffer.getChannelData(c), c);
          buffer = padded;
        }
        const gain = this.context.createGain();gain.gain.value = this.levels[track.id] ?? 0;
        gain.connect(this.context.destination);job.records.set(track.id, { buffer, gain });
      }
    };
    job.promise = Promise.all(Array.from({ length: Math.min(2, tracks.length) }, worker)).catch(error => {
      if (this.job === job) {
        job.controller.abort();this.job = null;
        for (const record of job.records.values()) record.gain.disconnect();
        job.records.clear();
      }
      throw error;
    });
    return job.promise;
  }
  setMix(levels) {
    this.levels = { ...levels };
    if (!this.context || !this.job) return;
    const now = this.context.currentTime;
    for (const [id, record] of this.job.records) {
      const gain = record.gain.gain;
      gain.cancelScheduledValues(now);gain.setValueAtTime(gain.value, now);
      gain.linearRampToValueAtTime(levels[id] ?? 0, now + 0.005);
    }
  }
  _position(anchor, now) {
    if (anchor.previous && now < anchor.when) return this._position(anchor.previous, now);
    const elapsed = Math.max(0, now - anchor.when);
    if (anchor.range) {
      const [a, b] = anchor.range;
      return a + ((anchor.offset - a + elapsed) % (b - a));
    }
    return Math.min(this.duration, anchor.offset + elapsed);
  }
  position() {
    return this.playing && this.anchor ? this._position(this.anchor, this.context.currentTime) : this.offset;
  }
  _retire(when) {
    for (const source of this.sources) { try { source.stop(when); } catch (_) { /* already ended */ } }
    this.sources = [];
    if (this.clock) { try { this.clock.stop(when); } catch (_) { /* already ended */ } this.clock = null; }
  }
  _schedule(offset, when, previous = null) {
    const generation = ++this.generation;
    this._retire(when);
    if (this.range) {
      const [a, b] = this.range;offset = a + (((offset - a) % (b - a)) + b - a) % (b - a);
    } else offset = Math.max(0, Math.min(offset, this.duration));
    this.anchor = { when, offset, range: this.range && [...this.range], previous };
    this.offset = offset;this.playing = true;
    try {
      for (const record of this.job.records.values()) {
        const source = this.context.createBufferSource();source.buffer = record.buffer;
        if (this.range) { source.loop = true;[source.loopStart, source.loopEnd] = this.range; }
        source.connect(record.gain);this.live.add(source);
        source.onended = () => { source.disconnect();this.live.delete(source); };
        this.sources.push(source);source.start(when, offset);
        if (!this.range) source.stop(when + Math.max(0, this.duration - offset));
      }
      // Completion follows the audio clock even if the UI thread is blocked.
      const clock = this.context.createBufferSource();clock.buffer = this.context.createBuffer(1, 1, this.context.sampleRate);
      clock.loop = true;clock.connect(this.context.destination);this.clock = clock;this.live.add(clock);
      clock.onended = () => {
        clock.disconnect();this.live.delete(clock);
        if (generation !== this.generation || !this.playing) return;
        this.playing = false;this.offset = this.duration;this.anchor = null;
        this._retire(this.context.currentTime);this.onEnded?.();
      };
      clock.start(when);
      if (!this.range) clock.stop(when + Math.max(0, this.duration - offset));
    } catch (error) { this.pause();throw error; }
  }
  start(offset) {
    if (!this.job?.records.size) throw new Error("试听音频尚未准备完成。");
    this._schedule(offset, this.context.currentTime + 0.04);
  }
  pause() {
    this.offset = this.position();this.playing = false;this.anchor = null;++this.generation;
    this._retire(this.context?.currentTime || 0);
    // Also stop graphs awaiting a scheduled loop transition.
    for (const source of this.live) { try { source.stop(this.context.currentTime); } catch (_) { /* ended */ } }
  }
  seek(offset) {
    offset = Math.max(0, Math.min(offset, this.duration));
    if (this.playing) { this.pause();this.start(offset); }
    else this.offset = offset;
  }
  setLoop(range) {
    const valid = range && range[0] >= 0 && range[1] <= this.duration + 0.001 && range[1] > range[0];
    const next = valid ? [range[0], Math.min(range[1], this.duration)] : null;
    if (JSON.stringify(next) === JSON.stringify(this.range)) return;
    const now = this.context?.currentTime || 0;const previous = this.anchor;
    const offset = previous ? this._position(previous, now + 0.04) : this.offset;this.range = next;
    if (this.playing) this._schedule(offset, now + 0.04, previous);
  }
}
const StemAudio = new StemTransport();
window.addEventListener("pagehide", () => StemAudio.dispose());
