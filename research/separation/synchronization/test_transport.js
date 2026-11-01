// Offline renders exercise the actual Web Audio graph, including its output samples.
async function transportRenderChecks() {
  const rate = 48000, checks = [];
  function fixture(seconds = 3) {
    const ctx = new OfflineAudioContext(2, seconds * rate, rate);
    const transport = new StemTransport(() => ctx);
    transport.context = ctx;transport.reset(1);
    const records = new Map();
    for (const [id, channel] of [['left', 0], ['right', 1]]) {
      const buffer = ctx.createBuffer(2, rate, rate);
      // Distinct output channels, identical known transient positions.
      for (let frame = 240; frame < rate; frame += 960) buffer.getChannelData(channel)[frame] = 0.5;
      const gain = ctx.createGain();gain.connect(ctx.destination);
      records.set(id, { buffer, gain });
    }
    transport.job = { records, controller: new AbortController(), promise: Promise.resolve() };
    transport.setMix({ left: 1, right: 1 });
    return { ctx, transport };
  }
  function assert(condition, message) { if (!condition) throw new Error(message); }
  function compare(buffer, ranges) {
    const a = buffer.getChannelData(0), b = buffer.getChannelData(1);
    let peakDifference = 0, activeSamples = 0;
    for (const [start, end] of ranges) for (let i = Math.ceil(start * rate); i < Math.floor(end * rate); i++) {
      peakDifference = Math.max(peakDifference, Math.abs(a[i] - b[i]));
      if (Math.abs(a[i]) > 0.1) activeSamples++;
    }
    assert(activeSamples > 10, 'Render must contain audible transients');
    assert(peakDifference === 0, 'Different output sample positions/amplitudes: ' + peakDifference);
    return { peak_difference: peakDifference, active_samples: activeSamples, offset_samples: 0 };
  }
  {
    const { ctx, transport: t } = fixture(3);
    t.setLoop([0.2, 0.45]);t.start(0.2);
    const generation = t.generation;
    const events = [
      [0.2, () => t.setMix({ left: 0, right: 1 })],
      [0.35, () => t.setMix({ left: 1, right: 1 })],
      [0.6, () => t.setLoop([0.3, 0.55])],
      [0.8, () => t.setMix({ left: 1, right: 0 })],
      [1.0, () => t.setMix({ left: 0, right: 1 })],
      [1.2, () => t.setMix({ left: 1, right: 1 })],
      [1.4, () => t.seek(0.4)],
      // Pause while the loop replacement is still scheduled in the future.
      [1.6, () => { t.setLoop([0.1, 0.35]);t.pause(); }],
      [1.75, () => t.start(0.25)],
      [1.9, () => t.setLoop(null)],
    ];
    const tasks = events.map(([when, fn]) => ctx.suspend(when).then(() => {
      fn();if (when === 0.35) assert(t.generation === generation, 'Mute recreated sources');
      return ctx.resume();
    }));
    const rendered = await ctx.startRendering();await Promise.all(tasks);
    const measured = compare(rendered, [[0.05, 0.19], [0.37, 0.79], [1.22, 1.59], [1.8, 2.6]]);
    const a = rendered.getChannelData(0), b = rendered.getChannelData(1);
    assert(a.slice(0.22 * rate, 0.34 * rate).every(v => v === 0), 'Mute did not silence channel');
    assert(b.slice(0.22 * rate, 0.34 * rate).some(v => v > 0.1), 'Solo also silenced other channel');
    assert(a.slice(1.61 * rate, 1.74 * rate).every(v => v === 0), 'Old scheduled graph leaked after pause');
    assert(b.slice(1.61 * rate, 1.74 * rate).every(v => v === 0), 'Old scheduled graph leaked after pause');
    checks.push({ name: 'rendered_mute_solo_seek_pause_loop_switch', ...measured });
  }
  {
    const { ctx, transport: t } = fixture(3);
    t.setLoop([0.2, 0.45]);t.start(0.2);
    // No animation frame or UI callback runs to restart any loop.
    const rendered = await ctx.startRendering();
    const measured = compare(rendered, [[0.05, 2.99]]);
    const a = rendered.getChannelData(0), period = 12000;
    for (let i = 0.3 * rate; i < 2.7 * rate; i++) assert(a[i] === a[i + period], 'Loop drifted');
    checks.push({ name: 'native_loop_without_ui_timer', ...measured, loop_period_samples: period });
  }
  {
    const { ctx, transport: t } = fixture(2);
    let ended = 0;t.onEnded = () => ended++;t.start(0.5);
    const rendered = await ctx.startRendering();
    await new Promise(resolve => setTimeout(resolve, 50));
    compare(rendered, [[0.05, 0.53]]);
    assert(rendered.getChannelData(0).slice(0.55 * rate).every(v => v === 0), 'Source continued past common end');
    assert(ended === 1 && !t.playing && t.position() === 1, 'Completion callback incorrect');
    checks.push({ name: 'shared_end_and_completion', ended });
  }
  return checks;
}
