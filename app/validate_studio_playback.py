"""Optional wall-clock playback soak test; never runs as part of unit tests."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

from test_studio_workbench import WorkbenchBrowserTest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 10:
        parser.error("--seconds must be at least 10")
    # Keep the CPU awake only for this opted-in long-running validation; the
    # display and user power settings remain unchanged.
    awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name == "nt" else False
    test = WorkbenchBrowserTest("test_ab_loop_with_audio_offset_and_nonzero_sync")
    test.setUpClass()
    test.setUp()
    started = time.monotonic()
    snapshots = []
    end_checks = []
    try:
        test.evaluate("""(() => {
          window.soak={frames:0,loops:0,lastView:null,started:performance.now()};
          const originalView=updateWorkbenchPlayback;
          updateWorkbenchPlayback=function(time,item,beat,node,index) {
            originalView(time,item,beat,node,index);
            soak.frames++;
            soak.lastView={time,chord:item?labelFor(item):'—',bar:beat?.bar??item?.bar??null,wall:performance.now()};
          };
          const originalGuard=enforcePlaybackRange;
          enforcePlaybackRange=function() {
            const before=audio.currentTime;
            originalGuard();
            if(audio.currentTime < before-1) soak.loops++;
          };
        })()""")
        phase = -1
        next_report = 60
        while time.monotonic() - started < args.seconds:
            elapsed = time.monotonic() - started
            current_phase = int(elapsed // 30)
            if current_phase != phase:
                phase = current_phase
                speed = [.5, .75, 1, 1.25, 1.5][phase % 5]
                sync = [-750, -200, 0, 250, 900][phase % 5]
                test.page.evaluate("""([speed,sync,index]) => {
                  audio.playbackRate=speed;syncMs=sync;displaySync();
                  setLoop(2,8.5);seekToAnalysis([2.25,3.3,7.9][index%3],true);
                  if(index%2) renderStable();
                }""", [speed, sync, phase])
                if phase and phase % 4 == 0:
                    test.evaluate("loopRange.enabled=false;auditionEnd=null;audio.currentTime=playbackEnd-.3;playAudio()")
                    test.page.wait_for_timeout(1700)
                    check = test.evaluate("({paused:audio.paused,time:audio.currentTime,end:playbackEnd,activeBars:document.querySelectorAll('.bar.active').length})")
                    end_checks.append(check)
                    test.evaluate("setLoop(2,8.5);seekToAnalysis(2.25,true)")
            test.page.wait_for_timeout(500)
            snapshots.append(test.evaluate("""(() => {
              const view=soak.lastView;
              let lag=view?analysisTime()-view.time:0;
              if(loopRange.enabled && lag < -3)lag+=loopRange.b-loopRange.a;
              return {wall:(performance.now()-soak.started)/1000,time:analysisTime(),viewTime:view?.time,
                lag,correctHighlight:!view||document.getElementById('currentChord').textContent===view.chord,
                activeBarCount:document.querySelectorAll('.bar.active').length,
                activeSpanCount:document.querySelectorAll('.span-cell.active').length,
                paused:audio.paused,rate:audio.playbackRate,sync:syncMs,loops:soak.loops,frames:soak.frames};
            })()"""))
            if elapsed >= next_report:
                print(f"Playback soak: {elapsed:.0f}s, {snapshots[-1]['loops']} loops, {snapshots[-1]['frames']} frames", flush=True)
                next_report += 60
        test.evaluate("audio.pause()")
        lags = [max(0,row["lag"]) for row in snapshots]
        ordered = sorted(lags)
        first = [max(0,row["lag"]) for row in snapshots if row["wall"] < 60]
        last = [max(0,row["lag"]) for row in snapshots if row["wall"] > args.seconds-60]
        maximum_gap = max((b["wall"]-a["wall"] for a,b in zip(snapshots,snapshots[1:])),default=0)
        failures = []
        if test.errors:
            failures.append("JavaScript exception")
        if any(not row["correctHighlight"] for row in snapshots):
            failures.append("Rendered chord differs from media-clock lookup")
        if any(row["activeBarCount"] > 1 or row["activeSpanCount"] > 1 for row in snapshots):
            failures.append("Active highlights accumulated")
        if any(not check["paused"] or abs(check["time"]-check["end"]) > .05 for check in end_checks):
            failures.append("Playback end was not clamped")
        if maximum_gap > 5:
            failures.append("Continuous playback interrupted by a sampling gap over five seconds")
        if any(row["paused"] for row in snapshots):
            failures.append("Playback unexpectedly paused during a loop")
        if any(row["lag"] >= 1 or row["lag"] < -.05 for row in snapshots):
            failures.append("Media clock and rendered view diverged")
        result = {
            "wall_seconds": round(time.monotonic()-started,3), "audio_offset":1.5,
            "requested_seconds":args.seconds,"max_sample_gap_seconds":round(maximum_gap,3),"idle_sleep_prevented":awake,
            "sample_count":len(snapshots), "speed_values":[.5,.75,1,1.25,1.5],
            "sync_values_ms":[-750,-200,0,250,900], "loops":snapshots[-1]["loops"],
            "rendered_frames":snapshots[-1]["frames"], "median_view_lag_ms":round(statistics.median(lags)*1000,3),
            "p95_view_lag_ms":round(ordered[int(.95*(len(ordered)-1))]*1000,3),
            "max_view_lag_ms":round(max(lags)*1000,3),
            "first_minute_median_lag_ms":round(statistics.median(first)*1000,3) if first else None,
            "last_minute_median_lag_ms":round(statistics.median(last)*1000,3) if last else None,
            "end_checks":end_checks,"javascript_errors":test.errors,"failures":failures,
            "pass":not failures,"samples":snapshots,
            "limitation":"Headless Edge media-clock/rendering check, not an acoustic output latency or sample-accurate loop measurement.",
        }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps({key:value for key,value in result.items() if key!="samples"},ensure_ascii=False),flush=True)
        return int(bool(failures))
    finally:
        test.context.close()
        test.tearDownClass()
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    raise SystemExit(main())
