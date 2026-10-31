const SOURCE_DATA = __ANALYSIS_JSON__;
    const DATA = cloneData(SOURCE_DATA);
    const audio = document.getElementById("audio");
    const playBtn = document.getElementById("playBtn");
    const seek = document.getElementById("seek");
    const sync = document.getElementById("sync");
    const syncValue = document.getElementById("syncValue");
    const barsEl = document.getElementById("bars");
    const eventList = document.getElementById("eventList");
    const currentChord = document.getElementById("currentChord");
    const currentTimeEl = document.getElementById("currentTime");
    const durationEl = document.getElementById("duration");
    const displayButtons = document.querySelectorAll("[data-display]");
    const editRange = document.getElementById("editRange");
    const chordInput = document.getElementById("chordInput");
    const saveChord = document.getElementById("saveChord");
    const clearChord = document.getElementById("clearChord");
    const resetChord = document.getElementById("resetChord");
    const splitHalf = document.getElementById("splitHalf");
    const splitBeats = document.getElementById("splitBeats");
    const split8 = document.getElementById("split8");
    const split16 = document.getElementById("split16");
    const mergeSame = document.getElementById("mergeSame");
    const copyCellBtn = document.getElementById("copyCell");
    const pasteCellBtn = document.getElementById("pasteCell");
    const copyBarBtn = document.getElementById("copyBar");
    const pasteBarBtn = document.getElementById("pasteBar");
    const copyStatus = document.getElementById("copyStatus");
    const scoreFile = document.getElementById("scoreFile");
    const scoreText = document.getElementById("scoreText");
    const scoreStart = document.getElementById("scoreStart");
    const scoreGrid = document.getElementById("scoreGrid");
    const scoreRepeat = document.getElementById("scoreRepeat");
    const applyScore = document.getElementById("applyScore");
    const clearScore = document.getElementById("clearScore");
    const scoreStatus = document.getElementById("scoreStatus");
    const exportLab = document.getElementById("exportLab");
    const exportJson = document.getElementById("exportJson");
    const resetAll = document.getElementById("resetAll");
    const suggestionList = document.getElementById("suggestionList");
    const pathProposalList = document.getElementById("pathProposalList");
    const editStatus = document.getElementById("editStatus");
    const playbackStart = Number(DATA.audio_offset || 0);
    const playbackDuration = Number(DATA.duration || 0);
    const playbackEnd = playbackStart + playbackDuration;
    const syncStorageKey = `chord-studio-sync:${DATA.source || DATA.audio_file}`;
    const legacyAutosaveStorageKey = `chord-studio-autosave:${DATA.autosave_namespace || DATA.analysis_id || DATA.source || DATA.audio_file}`;
    const autosaveStorageKey = `${legacyAutosaveStorageKey}:${baselineFingerprint(SOURCE_DATA)}`;
    const ROOT_TO_PC = { C: 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3, E: 4, F: 5, "F#": 6, Gb: 6, G: 7, "G#": 8, Ab: 8, A: 9, "A#": 10, Bb: 10, B: 11 };
    const ROOT_NAMES = Object.keys(ROOT_TO_PC).sort((a, b) => b.length - a.length);
    const PC_TO_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const PC_TO_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
    const MAJOR_ROMAN = { 0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII" };
    const MINOR_ROMAN = { 0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV", 6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "#VII" };
    const MAJOR_DEGREES = {
      "1": [0, ""], "2": [2, "m"], "3": [4, "m"], "4": [5, ""], "5": [7, ""], "6": [9, "m"], "7": [11, "dim"]
    };
    const MINOR_DEGREES = {
      "1": [0, "m"], "2": [2, "dim"], "3": [3, ""], "4": [5, "m"], "5": [7, ""], "6": [8, ""], "7": [10, ""]
    };
    let events = [];
    let beats = [];
    let eventIndexByIdentity = new Map();
    let eventIndexBySpan = new Map();
    let displayMode = DATA.display || "name";
    let syncMs = storedSyncMs();
    let rafId = null;
    let activeBar = null;
    let activeBeat = null;
    let activeEvent = null;
    let selectedEventIndex = 0;
    let selectedBeatTarget = null;
    let lastSelectedBarNumber = Number((DATA.bars || [])[0]?.bar || 1);
    let editClipboard = null;
    const dirtyEvents = new Set();

    function cloneData(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function baselineFingerprint(value) {
      // Include model provenance, confidence and nested timing: equal chord labels
      // alone do not identify an analysis version. The generated media URL is local.
      const baseline = { ...value };
      delete baseline.audio_src;
      delete baseline.audio_waveform;
      const signature = JSON.stringify(baseline);
      let first = 2166136261, second = 2654435761;
      for (let index = 0; index < signature.length; index++) {
        first = Math.imul(first ^ signature.charCodeAt(index),16777619);
        second = Math.imul(second ^ signature.charCodeAt(signature.length-1-index),2246822519);
      }
      return (first >>> 0).toString(16).padStart(8,"0") + (second >>> 0).toString(16).padStart(8,"0");
    }

    function autosaveDraft(reason = "edit") {
      recordHistory(reason);
      try {
        const payload = cloneData(DATA);
        payload.autosaved = true;
        payload.autosave_reason = reason;
        payload.autosaved_at = new Date().toISOString();
        localStorage.setItem(autosaveStorageKey, JSON.stringify(payload));
        if (editStatus) editStatus.textContent = `${editedEventCount()} 处修改 · 已保存 ${new Date().toLocaleTimeString()}`;
        document.getElementById("sessionStatus").textContent = audio.error ? "音频未连接 · 标注已自动保存" : "修改已自动保存在此浏览器";
      } catch (error) {
        if (editStatus) editStatus.textContent = "本地保存空间不足，请导出 JSON 保存修改";
        toast("浏览器未能保存草稿，请导出 JSON 文件保存本次修改。", 7000);
      }
    }

    function clearAutosaveDraft() {
      try {
        localStorage.removeItem(autosaveStorageKey);
        localStorage.removeItem(legacyAutosaveStorageKey);
      } catch (error) {
      }
    }

    function restoreAutosaveDraft() {
      try {
        const current = localStorage.getItem(autosaveStorageKey);
        const raw = current || localStorage.getItem(legacyAutosaveStorageKey);
        if (!raw) return false;
        const draft = JSON.parse(raw);
        if (!isCompatibleAnalysis(draft)) return false;
        const count = Number(draft.corrected_event_count || 0) || (draft.events || []).filter((event) => event.edited).length;
        const stamp = draft.autosaved_at ? new Date(draft.autosaved_at).toLocaleString() : "未知时间";
        const legacyWarning = current ? "" : "这是旧版工作台的草稿，可能来自该歌曲的另一份分析。";
        if (!window.confirm(`发现自动保存修改（${stamp}，约 ${count} 处）。${legacyWarning}是否恢复？`)) return false;
        for (const key of Object.keys(DATA)) delete DATA[key];
        Object.assign(DATA, draft);
        if (!current) {
          try { localStorage.setItem(autosaveStorageKey,raw); localStorage.removeItem(legacyAutosaveStorageKey); } catch (error) {}
        }
        if (editStatus) editStatus.textContent = "已恢复自动保存";
        return true;
      } catch (error) {
        if (editStatus) editStatus.textContent = "自动保存恢复失败";
        return false;
      }
    }

    function refreshDerived() {
      events = [...(DATA.events || [])].sort((left, right) => Number(left.start || 0) - Number(right.start || 0) || Number(left.end || 0) - Number(right.end || 0));
      beats = (DATA.bars || [])
        .flatMap((bar) => (bar.beats || []).map((beat) => ({ ...beat, bar: bar.bar })))
        .sort((left, right) => Number(left.start || 0) - Number(right.start || 0) || Number(left.end || 0) - Number(right.end || 0));
      DATA.events = events;
      eventIndexByIdentity = new Map(events.map((event, index) => [event, index]));
      eventIndexBySpan = new Map(events.map((event, index) => [spanIdentity(event), index]));
    }

    function spanIdentity(item) {
      return `${Number(item.start).toFixed(6)}|${Number(item.end).toFixed(6)}|${item.chord_display || item.chord || "N"}`;
    }

    function storedSyncMs() {
      try {
        const value = Number(localStorage.getItem(syncStorageKey) || 0);
        return Number.isFinite(value) ? clamp(Math.round(value / 10) * 10,-2000,2000) : 0;
      } catch (error) {
        return 0;
      }
    }

    function fmt(seconds) {
      const total = Math.max(0, Math.floor(seconds || 0));
      const m = Math.floor(total / 60);
      const s = total % 60;
      const h = Math.floor(m / 60);
      const mm = m % 60;
      if (h > 0) return `${String(h).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
      return `${String(mm).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function localPlaybackTime() {
      return clamp(Number(audio.currentTime || 0) - playbackStart, 0, playbackDuration);
    }

    function analysisTime() {
      const local = localPlaybackTime();
      return clamp(local + syncMs / 1000, 0, playbackDuration);
    }

    function displaySync() {
      sync.value = String(syncMs);
      syncValue.textContent = `${syncMs > 0 ? "+" : ""}${syncMs}ms`;
    }

    function showPlaybackError(error) {
      if (error?.name === "AbortError") return;
      document.getElementById("sessionStatus").textContent = "音频未连接，请选择本地音频";
      toast("音频暂时无法播放。点击「选择音频」重新连接，标注仍可继续。", 6500);
    }

    function playAudio() {
      const result = audio.play();
      if (result && typeof result.catch === "function") {
        result.catch(showPlaybackError);
      }
      return result;
    }

    function renderStable() {
      const scrollX = window.scrollX;
      const scrollY = window.scrollY;
      const currentTime = Number(audio.currentTime || 0);
      const wasPlaying = !audio.paused && !audio.ended;
      render();
      const restore = () => {
        window.scrollTo(scrollX, scrollY);
        if (Number.isFinite(currentTime) && Math.abs(Number(audio.currentTime || 0) - currentTime) > 0.25) {
          audio.currentTime = currentTime;
        }
        if (wasPlaying && audio.paused) {
          playAudio();
        }
      };
      restore();
      requestAnimationFrame(restore);
    }

    function parseChord(label) {
      const raw = String(label || "").trim().replace(/♯/g, "#").replace(/♭/g, "b");
      if (raw.toUpperCase() === "X") {
        return { chord: "X", chordDisplay: "X", roman: "X", romanDisplay: "X", bassRoot: "N", bassConfidence: 0 };
      }
      if (!raw || ["N", "NC"].includes(raw.toUpperCase())) {
        return { chord: "N", chordDisplay: "N", roman: "N", romanDisplay: "N", bassRoot: "N", bassConfidence: 0 };
      }
      const parts = raw.split("/");
      const body = parts[0].trim();
      let bassRoot = parts[1] ? parts[1].trim() : "N";
      if (parts.length > 2 || (parts.length === 2 && !parts[1].trim())) return null;
      if (parts.length === 2 && bassRoot === "N") return null;
      let root = null;
      let suffix = "";
      for (const candidate of ROOT_NAMES) {
        if (body.startsWith(candidate)) {
          root = candidate;
          suffix = body.slice(candidate.length);
          break;
        }
      }
      if (!root) {
        return null;
      }
      if (bassRoot !== "N" && !Object.hasOwn(ROOT_TO_PC, bassRoot)) {
        // Harte slash degrees are relative to the chord root. The existing
        // editor's uncolonized degree input remains relative to the key.
        const relativeBass = body.includes(":") && bassRoot.match(/^([b#]*)(1[0-3]|[1-9])$/);
        if (!relativeBass) return null;
        const accidental = [...relativeBass[1]].reduce((sum, symbol) => sum + (symbol === "#" ? 1 : -1), 0);
        const semitone = [0, 2, 4, 5, 7, 9, 11][(Number(relativeBass[2])-1)%7];
        bassRoot = noteName(ROOT_TO_PC[root] + semitone + accidental);
      }
      if (suffix.startsWith(":")) {
        const originalQuality = suffix.slice(1);
        const quality = /^M(?:7|9|11|13)?$/.test(originalQuality) ? `maj${originalQuality.slice(1)}` : originalQuality.toLowerCase();
        const map = { maj: "", major: "", min: "m", minor: "m", "7": "7", maj7: "maj7", min7: "m7", maj6: "6", min6: "m6", min9: "m9", min11: "m11", min13: "m13", minmaj7: "mMaj7", hdim7: "m7b5", dim: "dim", aug: "aug", sus: "sus4", sus4: "sus4", sus2: "sus2" };
        suffix = map[quality] ?? quality;
      }
      const namedQuality = /^(?:mMaj|m|maj|M|min|dim|aug|sus[24]?|add[2469]|[+°ø])?(?:[5679]|11|13)?(?:sus[24]|add(?:[2469]|11|13)|\(?[#b](?:5|9|11|13)\)?)*$/;
      const intervalQuality = /^(?:maj|min|dim|aug|sus[24]|[5679]|11|13|maj[679]|min[679]|maj13|min11|min13|minmaj7|dim7|hdim7)?\(\*?[b#]*(?:1[0-3]|[1-9])(?:,\*?[b#]*(?:1[0-3]|[1-9]))*\)$/;
      if (!namedQuality.test(suffix) && !intervalQuality.test(suffix)) return null;
      if (suffix === "M") suffix = "";
      else if (/^M(?:7|9|11|13)$/.test(suffix)) suffix = `maj${suffix.slice(1)}`;
      const chord = `${root}${suffix}`;
      const roman = chordToRoman(chord);
      const bassRoman = bassRoot !== "N" ? rootToRoman(bassRoot) : "";
      const chordDisplay = bassRoot !== "N" ? `${chord}/${bassRoot}` : chord;
      const romanDisplay = bassRoman ? `${roman}/${bassRoman}` : roman;
      return {
        chord: chordDisplay,
        chordDisplay,
        roman: romanDisplay,
        romanDisplay,
        bassRoot,
        bassConfidence: bassRoot !== "N" ? 1 : 0,
      };
    }

    function rootToRoman(rootName) {
      const key = parseChordRoot(editingKey());
      const root = parseChordRoot(rootName || "");
      if (!key || !root) return rootName || "";
      const map = key.suffix === "m" ? MINOR_ROMAN : MAJOR_ROMAN;
      return map[(root.pc - key.pc + 12) % 12] || rootName;
    }

    function chordToRoman(chord) {
      if (chord === "N" || chord === "X") return chord;
      const key = parseChordRoot(editingKey());
      const parsed = parseChordRoot(chord || "");
      if (!key || !parsed) return chord;
      const map = key.suffix === "m" ? MINOR_ROMAN : MAJOR_ROMAN;
      const label = `${map[(parsed.pc - key.pc + 12) % 12] || parsed.root}${parsed.suffix}`;
      const bass = String(chord).split("/")[1];
      return bass && Object.hasOwn(ROOT_TO_PC, bass) ? `${label}/${rootToRoman(bass)}` : label;
    }

    function parseChordRoot(label) {
      const text = String(label || "").split("/", 1)[0].trim();
      for (const candidate of ROOT_NAMES) {
        if (text.startsWith(candidate)) {
          return { root: candidate, pc: ROOT_TO_PC[candidate], suffix: text.slice(candidate.length) };
        }
      }
      return null;
    }

    function noteName(pc) {
      const key = String(editingKey());
      const names = key.includes("b") ? PC_TO_FLAT : PC_TO_SHARP;
      return names[((pc % 12) + 12) % 12];
    }

    function degreeTokenToChord(token) {
      const text = String(token || "").trim();
      if (!text) return null;
      if (parseChord(text)) return text;

      const numeric = text.match(/^([b#]?)([1-7])([A-Za-z0-9+°ø#b]*)$/);
      if (numeric) {
        return degreeToChord(numeric[2], numeric[1], numeric[3] || "");
      }

      const roman = text.match(/^([b#]?)(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)(.*)$/);
      if (!roman) return null;
      const degreeMap = { I: "1", II: "2", III: "3", IV: "4", V: "5", VI: "6", VII: "7" };
      const romanBody = roman[2];
      const degree = degreeMap[romanBody.toUpperCase()];
      const impliedMinor = romanBody === romanBody.toLowerCase();
      let suffix = roman[3] || "";
      let quality = impliedMinor ? "m" : "";
      if (suffix.startsWith("m") && !suffix.startsWith("maj")) {
        quality = "m";
        suffix = suffix.slice(1);
      }
      return degreeToChord(degree, roman[1], suffix, quality);
    }

    function degreeToChord(degree, accidental = "", suffix = "", forceQuality = null) {
      const key = parseChordRoot(editingKey());
      if (!key) return null;
      const table = key.suffix === "m" ? MINOR_DEGREES : MAJOR_DEGREES;
      const spec = table[degree];
      if (!spec) return null;
      const accidentalOffset = accidental === "b" ? -1 : accidental === "#" ? 1 : 0;
      const root = noteName(key.pc + spec[0] + accidentalOffset);
      let quality = forceQuality == null ? spec[1] : forceQuality;
      const lowered = String(suffix || "").toLowerCase();
      if (/^maj(?:7|9|11|13)$/.test(suffix)) {
        quality = "";
      } else if (/^m(?:7|9|11|13)$/.test(suffix)) {
        quality = "m";
        suffix = suffix.slice(1);
      } else if (lowered === "maj" || lowered === "major") {
        quality = "";
        suffix = "";
      } else if (lowered === "min" || lowered === "minor") {
        quality = "m";
        suffix = "";
      } else if (suffix === "m") {
        quality = "m";
        suffix = "";
      }
      if (quality === "dim" && suffix === "7") return `${root}dim7`;
      return `${root}${quality}${suffix}`;
    }

    function parseEditableChord(label) {
      const text = String(label || "").trim();
      if (!text) return parseChord("N");
      const parts = text.split("/");
      if (parts.length > 2 || (parts.length === 2 && !parts[1].trim())) return null;
      const direct = parseChord(text);
      if (direct) return direct;
      const mainDirect = parseChord(parts[0]);
      const chordName = mainDirect ? mainDirect.chord : degreeTokenToChord(parts[0]);
      if (!chordName) return null;
      let bassName = "";
      if (parts.length > 1 && parts[1].trim()) {
        bassName = degreeTokenToBass(parts[1].trim()) || parts[1].trim();
      }
      return parseChord(bassName ? `${chordName}/${bassName}` : chordName);
    }

    function degreeTokenToBass(token) {
      const text = String(token || "").trim();
      if (!text) return null;
      const root = parseChordRoot(text);
      if (root) return root.root;
      const numeric = text.match(/^([b#]?)([1-7])$/);
      if (numeric) {
        return degreeRootName(numeric[2], numeric[1]);
      }
      const roman = text.match(/^([b#]?)(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)$/);
      if (!roman) return null;
      const degreeMap = { I: "1", II: "2", III: "3", IV: "4", V: "5", VI: "6", VII: "7" };
      return degreeRootName(degreeMap[roman[2].toUpperCase()], roman[1]);
    }

    function degreeRootName(degree, accidental = "") {
      const key = parseChordRoot(editingKey());
      if (!key) return null;
      const table = key.suffix === "m" ? MINOR_DEGREES : MAJOR_DEGREES;
      const spec = table[degree];
      if (!spec) return null;
      const accidentalOffset = accidental === "b" ? -1 : accidental === "#" ? 1 : 0;
      return noteName(key.pc + spec[0] + accidentalOffset);
    }

    function parseScoreText(text) {
      const bars = [];
      const lines = String(text || "")
        .replace(/｜/g, "|")
        .replace(/[–—]/g, "-")
        .split(/\r?\n/);
      for (let line of lines) {
        line = line.trim();
        if (!line || /^\[[^\]]+\]$/.test(line)) continue;
        line = line.replace(/\|\|/g, "|").replace(/(^|\s):|:(\s|$)/g, " ");
        if (line.includes("|")) {
          const parts = line.split("|").map((part) => part.trim()).filter(Boolean);
          for (const part of parts) {
            const tokens = scoreTokensForBar(part);
            if (tokens.length) bars.push(tokens);
          }
        } else {
          const tokens = scoreTokensForBar(line);
          if (tokens.length === 1 && /^[1-7]{2,}$/.test(tokens[0])) {
            for (const digit of tokens[0]) bars.push([digit]);
          } else if (tokens.length > 1) {
            for (const token of tokens) bars.push([token]);
          } else if (tokens.length) {
            bars.push(tokens);
          }
        }
      }
      return bars;
    }

    function scoreTokensForBar(text) {
      const compact = String(text || "").trim();
      if (!compact || compact === "%" || compact === ".") return [];
      if (/^[1-7]{2,}$/.test(compact)) return [compact];
      if (/^[b#]?[1-7](?:m|maj|dim|aug|sus|add|[0-9])*([\-–—][b#]?[1-7](?:m|maj|dim|aug|sus|add|[0-9])*)+$/.test(compact)) {
        return compact.split(/[\-–—]/).filter(Boolean);
      }
      return compact
        .replace(/[，,]/g, " ")
        .split(/\s+/)
        .map((token) => token.trim().replace(/^\(|\)$/g, ""))
        .filter(Boolean);
    }

    function applyChordFields(item, parsed, confidence = 1, edited = true) {
      item.chord = parsed.chord;
      item.chord_display = parsed.chordDisplay;
      item.roman = parsed.roman;
      item.roman_display = parsed.romanDisplay;
      item.bass_root = parsed.bassRoot;
      item.bass_confidence = parsed.bassConfidence;
      item.confidence = confidence;
      if (edited) item.edited = true;
      else delete item.edited;
      if (edited) {
        delete item.extension_refined;
        delete item.extension_refinement;
      }
    }

    function overlaps(left, right) {
      return left.start < right.end && right.start < left.end;
    }

    function sameSpan(left, right) {
      return Math.abs(Number(left.start) - Number(right.start)) < 0.0001
        && Math.abs(Number(left.end) - Number(right.end)) < 0.0001;
    }

    function updateCoveredItems(target, parsed, confidence = 1, edited = true) {
      applyChordFields(target, parsed, confidence, edited);
      for (const bar of DATA.bars || []) {
        const spans = [];
        for (const span of bar.spans || []) {
          if (!overlaps(span,target)) { spans.push(span); continue; }
          const left = Math.max(Number(span.start),Number(target.start));
          const right = Math.min(Number(span.end),Number(target.end));
          if (Number(span.start) < left-.0001) spans.push({ ...span,end:left });
          const changed = { ...span,start:left,end:right };
          applyChordFields(changed,parsed,confidence,edited);
          spans.push(changed);
          if (Number(span.end) > right+.0001) spans.push({ ...span,start:right });
        }
        bar.spans = spans;
        for (const beat of bar.beats || []) {
          if (overlaps(beat, target)) {
            const covering = activeFor(beatCenter(beat));
            if (covering) applyChordFields(beat, parsedFromItem(covering), Number(covering.confidence ?? 1), Boolean(covering.edited));
          }
        }
      }
    }

    function editedEventCount() {
      return events.reduce((count, event) => count + (event.edited ? 1 : 0), 0);
    }

    function barByNumber(barNumber) {
      return (DATA.bars || []).find((bar) => Number(bar.bar) === Number(barNumber)) || null;
    }

    function beatByNumber(barNumber, beatNumber, source = DATA) {
      const bar = (source.bars || []).find((item) => Number(item.bar) === Number(barNumber));
      if (!bar) return null;
      return (bar.beats || []).find((beat) => Number(beat.beat) === Number(beatNumber)) || null;
    }

    function beatCenter(beat) {
      return (Number(beat.start) + Number(beat.end)) / 2;
    }

    function barDuration(bar) {
      return Math.max(Number(bar.end || 0) - Number(bar.start || 0), 0.001);
    }

    function barFraction(bar, time) {
      return clamp((Number(time || 0) - Number(bar.start || 0)) / barDuration(bar), 0, 1);
    }

    function beatPositionAtTime(bar, time) {
      const beatsInBar = [...(bar.beats || [])].sort((left, right) => Number(left.start) - Number(right.start));
      if (!beatsInBar.length) return 1 + barFraction(bar, time) * beatCountForBar(bar);
      for (let index = 0; index < beatsInBar.length; index += 1) {
        const beat = beatsInBar[index];
        if (time >= Number(beat.start) && time <= Number(beat.end)) {
          const span = Math.max(Number(beat.end) - Number(beat.start), 0.001);
          return Number(beat.beat || index + 1) + clamp((time - Number(beat.start)) / span, 0, 1);
        }
      }
      return Number(beatsInBar[beatsInBar.length - 1].beat || beatsInBar.length);
    }

    function sixteenthLabel(bar, time) {
      const position = beatPositionAtTime(bar, Number(time || 0));
      const beat = Math.max(1, Math.floor(position));
      const fraction = clamp(position - beat, 0, 0.999);
      const divisions = 16 / meterDenominatorForBar(bar);
      if (divisions < 1) return `${beat}拍`;
      const sixteenth = Math.floor(fraction * divisions) + 1;
      return `${beat}.${sixteenth}/16`;
    }

    function spanUnit(beatStart, beatEnd, bar) {
      const beatsInBar = bar.beats || [];
      const firstBeat = beatsInBar[0]?.beat ?? 1;
      const lastBeat = beatsInBar[beatsInBar.length - 1]?.beat ?? beatCountForBar(bar);
      return beatStart <= firstBeat && beatEnd >= lastBeat ? "bar" : "beat";
    }

    function eventFromBeats(bar, beatGroup, parsed, confidence = 1, edited = true) {
      const first = beatGroup[0];
      const last = beatGroup[beatGroup.length - 1];
      const event = {
        bar: bar.bar,
        unit: spanUnit(first.beat, last.beat, bar),
        beat_start: first.beat,
        beat_end: last.beat,
        start: first.start,
        end: last.end,
        chord: "N",
        chord_display: "N",
        roman: "N",
        roman_display: "N",
        bass_root: "N",
        bass_confidence: 0,
        confidence,
      };
      applyChordFields(event, parsed, confidence, edited);
      return event;
    }

    function eventFromTimeRange(bar, start, end, parsed, confidence = 1, edited = true, unit = "subbeat") {
      const beatStart = beatPositionAtTime(bar, start);
      const beatEnd = beatPositionAtTime(bar, Math.max(start, end - 0.001));
      const event = {
        bar: bar.bar,
        unit,
        beat_start: Number(beatStart.toFixed(2)),
        beat_end: Number(beatEnd.toFixed(2)),
        start,
        end,
        chord: "N",
        chord_display: "N",
        roman: "N",
        roman_display: "N",
        bass_root: "N",
        bass_confidence: 0,
        confidence,
        position: sixteenthLabel(bar, start),
      };
      applyChordFields(event, parsed, confidence, edited);
      return event;
    }

    function eventFromExistingBeats(bar, beatGroup) {
      const first = beatGroup[0];
      const last = beatGroup[beatGroup.length - 1];
      const confidence = beatGroup.reduce((sum, beat) => sum + Number(beat.confidence || 0), 0) / Math.max(beatGroup.length, 1);
      return {
        bar: bar.bar,
        unit: spanUnit(first.beat, last.beat, bar),
        beat_start: first.beat,
        beat_end: last.beat,
        start: first.start,
        end: last.end,
        chord: first.chord || "N",
        chord_display: first.chord_display || first.chord || "N",
        roman: first.roman || "N",
        roman_display: first.roman_display || first.roman || "N",
        bass_root: first.bass_root || "N",
        bass_confidence: first.bass_confidence || 0,
        confidence,
        edited: beatGroup.some((beat) => beat.edited),
      };
    }

    function beatChordKey(beat) {
      return [
        beat.chord || "N",
        beat.chord_display || beat.chord || "N",
        beat.roman_display || beat.roman || "N",
        beat.bass_root || "N",
      ].join("|");
    }

    function rebuildBarSpansFromBeats(bar) {
      const sortedBeats = [...(bar.beats || [])].sort((left, right) => left.beat - right.beat);
      const groups = [];
      let current = [];
      for (const beat of sortedBeats) {
        if (!current.length || beatChordKey(current[current.length - 1]) === beatChordKey(beat)) {
          current.push(beat);
        } else {
          groups.push(current);
          current = [beat];
        }
      }
      if (current.length) groups.push(current);
      const newEvents = groups.map((group) => eventFromExistingBeats(bar, group));
      replaceBarSpans(bar, newEvents);
      return newEvents;
    }

    function partitionBeats(beatsInBar, segmentCount) {
      const beatsToUse = beatsInBar || [];
      if (!beatsToUse.length || segmentCount <= 0) return [];
      const count = Math.min(segmentCount, beatsToUse.length);
      const groups = [];
      for (let index = 0; index < count; index += 1) {
        const startIndex = Math.floor((index * beatsToUse.length) / count);
        const endIndex = Math.max(startIndex, Math.floor(((index + 1) * beatsToUse.length) / count) - 1);
        groups.push(beatsToUse.slice(startIndex, endIndex + 1));
      }
      return groups.filter((group) => group.length);
    }

    function replaceBarSpans(bar, newEvents) {
      const sorted = [...newEvents].sort((left, right) => left.start - right.start);
      bar.spans = sorted.map((event) => ({ ...event }));
      const preserved = [];
      for (const event of events) {
        if (!overlaps(event,bar)) { preserved.push(event); continue; }
        if (Number(event.start) < Number(bar.start)-.0001) preserved.push({ ...event,end:Number(bar.start) });
        if (Number(event.end) > Number(bar.end)+.0001) {
          const nextBar = (DATA.bars || []).find((candidate) => Number(bar.end) >= Number(candidate.start) && Number(bar.end) < Number(candidate.end));
          preserved.push({ ...event,start:Number(bar.end),bar:nextBar?.bar ?? event.bar,beat_start:nextBar?.beats?.length ? Number(beatPositionAtTime(nextBar,Number(bar.end)).toFixed(3)) : null });
        }
      }
      DATA.events = [
        ...preserved,
        ...sorted.map((event) => ({ ...event })),
      ].sort((left, right) => left.start - right.start || left.end - right.end);
      for (const beat of bar.beats || []) {
        const center = beatCenter(beat);
        const covering = sorted.find((event) => center >= event.start && center < event.end) || sorted[sorted.length - 1];
        if (covering) {
          applyChordFields(beat, {
            chord: covering.chord,
            chordDisplay: covering.chord_display,
            roman: covering.roman,
            romanDisplay: covering.roman_display,
            bassRoot: covering.bass_root,
            bassConfidence: covering.bass_confidence,
          }, covering.confidence, Boolean(covering.edited));
        }
      }
      refreshDerived();
    }

    function segmentsForBar(bar, chordLabels, gridMode = "auto") {
      const beatsInBar = bar.beats || [];
      const labels = chordLabels.filter(Boolean);
      if (!labels.length || !beatsInBar.length) return [];
      const parsed = withKeyAtTime(bar.start, () => labels.map((label) => parseEditableChord(label))).filter(Boolean);
      if (!parsed.length) return [];
      let segmentCount = parsed.length;
      if (gridMode === "bar") segmentCount = 1;
      else if (gridMode === "half") segmentCount = Math.min(Math.max(parsed.length, 2), beatsInBar.length);
      else if (gridMode === "beat") segmentCount = beatsInBar.length;
      else if (gridMode === "sixteenth") segmentCount = Math.min(Math.max(parsed.length, 1), beatCountForBar(bar) * 16 / meterDenominatorForBar(bar));
      else if (parsed.length === 1) segmentCount = 1;
      else if (parsed.length === 2) segmentCount = 2;
      else segmentCount = Math.min(parsed.length, beatsInBar.length);

      if (gridMode === "sixteenth") {
        const start = Number(bar.start || 0);
        const end = Number(bar.end || start);
        const duration = Math.max(end - start, 0.001);
        return parsed.slice(0, segmentCount).map((chord, index) => {
          const segmentStart = start + (duration * index) / segmentCount;
          const segmentEnd = start + (duration * (index + 1)) / segmentCount;
          return eventFromTimeRange(bar, segmentStart, segmentEnd, chord, 1, true);
        });
      }

      const groups = partitionBeats(beatsInBar, segmentCount);
      return groups.map((group, index) => {
        const chord = parsed[Math.min(index, parsed.length - 1)];
        return eventFromBeats(bar, group, chord, 1, true);
      });
    }

    function splitSelectedEvent(mode) {
      if (!canEditTimeline()) return;
      const target = events[selectedEventIndex];
      const bar = barByNumber(target?.bar);
      if (!target || !bar) return;
      if (Number(target.end) > Number(bar.end)+.0001 || ["uncovered","incomplete"].includes(bar.grid_status)) { toast("该片段跨越小节或拍网格尚不完整。请先核对拍点，当前可修改和弦标签。"); return; }
      const parsed = parsedFromItem(target);
      const start = Number(target.start), end = Number(target.end);
      const cuts = [start, end];
      if (mode === "half") cuts.push((start + end) / 2);
      else {
        const subdivisions = mode === "eighth" ? 8 / meterDenominatorForBar(bar) : mode === "sixteenth" ? 16 / meterDenominatorForBar(bar) : 1;
        for (const beat of bar.beats || []) {
          if (subdivisions < 1 && (Number(beat.beat)-1) % Math.round(1/subdivisions) !== 0) continue;
          for (let index = 0; index < subdivisions; index += 1) {
            const time = Number(beat.start) + (Number(beat.end) - Number(beat.start)) * index / Math.max(1,subdivisions);
            if (time > start + .005 && time < end - .005) cuts.push(time);
          }
        }
      }
      const sorted = [...new Set(cuts)].sort((a,b) => a-b);
      if (sorted.length < 3) { toast("当前片段内没有可切分的拍点。"); return; }
      const pieces = sorted.slice(0,-1).map((time,index) => eventFromTimeRange(bar,time,sorted[index+1],parsed,1,true,mode));
      replaceBarSpans(bar,[...(bar.spans || []).filter((span) => !sameSpan(span,target)),...pieces]);
      selectedBeatTarget = null;
      selectedEventIndex = events.findIndex((event) => sameSpan(event,pieces[0]));
      renderStable(); selectEvent(selectedEventIndex); updateActive(); autosaveDraft("split");
    }

    function mergeAdjacentSameChords() {
      if (!canEditTimeline()) return;
      const merged = [];
      for (const bar of DATA.bars || []) {
        const spans = [...(bar.spans || [])].sort((left, right) => left.start - right.start);
        const nextSpans = [];
        for (const span of spans) {
          const prev = nextSpans[nextSpans.length - 1];
          if (
            prev
            && prev.chord_display === span.chord_display
            && Math.abs(prev.end - span.start) < 0.04
          ) {
            prev.end = span.end;
            prev.beat_end = span.beat_end;
            prev.unit = spanUnit(prev.beat_start, prev.beat_end, bar);
            prev.confidence = Math.max(prev.confidence || 0, span.confidence || 0);
            prev.edited = Boolean(prev.edited || span.edited);
          } else {
            nextSpans.push({ ...span });
          }
        }
        replaceBarSpans(bar, nextSpans);
        merged.push(...nextSpans);
      }
      DATA.events = merged.sort((left, right) => left.start - right.start || left.end - right.end);
      refreshDerived();
      renderStable();
      selectEvent(Math.min(selectedEventIndex, events.length - 1));
      updateActive();
      autosaveDraft("merge");
    }

    function applyScoreToTimeline() {
      if (!canEditTimeline()) return;
      const pattern = parseScoreText(scoreText.value);
      if (!pattern.length) {
        scoreStatus.textContent = "未识别到和弦";
        return;
      }
      const startBar = Math.max(1, Number(scoreStart.value || 1));
      const repeat = Boolean(scoreRepeat.checked);
      const gridMode = scoreGrid.value || "auto";
      const targetBars = (DATA.bars || []).filter((bar) => Number(bar.bar) >= startBar && bar.grid_status !== "uncovered");
      let applied = 0, skipped = 0;
      for (let index = 0; index < targetBars.length; index += 1) {
        if (!repeat && index >= pattern.length) break;
        const spec = pattern[index % pattern.length];
        const bar = targetBars[index];
        if (bar.grid_status === "incomplete") { skipped += 1; continue; }
        const spans = segmentsForBar(bar, spec, gridMode);
        if (!spans.length) continue;
        replaceBarSpans(bar, spans);
        applied += 1;
      }
      if (!applied) {
        scoreStatus.textContent = "没有可应用的小节";
        return;
      }
      refreshDerived();
      selectedEventIndex = events.findIndex((event) => Number(event.bar) >= startBar);
      if (selectedEventIndex < 0) selectedEventIndex = 0;
      renderStable();
      selectEvent(selectedEventIndex);
      updateActive();
      scoreStatus.textContent = `${pattern.length} 小节谱，已铺 ${applied} 小节${skipped ? `，跳过 ${skipped} 个拍点不完整小节（后续小节仍按原序号对应）` : ""}`;
      autosaveDraft("score");
    }

    function loadScoreFile(file) {
      if (!file) return;
      const reader = new FileReader();
      reader.addEventListener("load", () => {
        const text = String(reader.result || "");
        if (file.name.toLowerCase().endsWith(".json")) {
          try {
            const json = JSON.parse(text);
            const rows = json.events || json.chords || json.annotations || [];
            scoreText.value = rows.map((row) => row.chord_display || row.chord || row.label || row.name || "").filter(Boolean).join(" ");
          } catch (error) {
            scoreText.value = text;
          }
        } else if (file.name.toLowerCase().endsWith(".lab")) {
          scoreText.value = text.split(/\r?\n/).map((line) => line.trim().split(/\s+/)[2] || "").filter(Boolean).join(" ");
        } else if (file.name.toLowerCase().endsWith(".csv")) {
          scoreText.value = text.split(/\r?\n/).slice(1).map((line) => {
            const cols = line.split(",");
            return cols[2] || cols[cols.length - 1] || "";
          }).filter(Boolean).join(" ");
        } else {
          scoreText.value = text;
        }
        scoreStatus.textContent = `${file.name}`;
      });
      reader.readAsText(file, "utf-8");
    }

    function updateEditPanel() {
      if (!events.length) {
        editRange.textContent = "--";
        chordInput.value = "N";
        return;
      }
      document.querySelectorAll(".event.selected").forEach((node) => node.classList.remove("selected"));
      document.querySelectorAll(".beat.selected").forEach((node) => node.classList.remove("selected"));
      document.querySelectorAll(".bar.selected").forEach((node) => node.classList.remove("selected"));
      document.querySelectorAll(".span-cell.selected").forEach((node) => node.classList.remove("selected"));
      chordInput.removeAttribute("aria-invalid");
      if (selectedBeatTarget) {
        const beat = beatByNumber(selectedBeatTarget.bar, selectedBeatTarget.beat);
        if (beat) {
          lastSelectedBarNumber = Number(selectedBeatTarget.bar);
          const eventIndex = events.findIndex((event) => beatCenter(beat) >= event.start && beatCenter(beat) < event.end);
          selectedEventIndex = eventIndex >= 0 ? eventIndex : selectedEventIndex;
          editRange.textContent = `第 ${selectedBeatTarget.bar} 小节 · 第 ${selectedBeatTarget.beat} 拍 · ${preciseTime(beat.start)}`;
          chordInput.value = beat.chord_display || beat.chord || "N";
          editStatus.textContent = `${editedEventCount()} 修改`;
          const selectedBeat = document.querySelector(`.beat[data-bar="${selectedBeatTarget.bar}"][data-beat="${selectedBeatTarget.beat}"]`);
          if (selectedBeat) selectedBeat.classList.add("selected");
          const selectedBar = document.querySelector(`.bar[data-bar="${selectedBeatTarget.bar}"]`);
          if (selectedBar) selectedBar.classList.add("selected");
          const selectedRow = document.querySelector(`.event[data-event="${selectedEventIndex}"]`);
          if (selectedRow) selectedRow.classList.add("selected");
          updateSelectionDetails(beat);
          return;
        }
        selectedBeatTarget = null;
      }
      selectedEventIndex = clamp(selectedEventIndex, 0, events.length - 1);
      const item = events[selectedEventIndex];
      if (item) lastSelectedBarNumber = Number(item.bar ?? lastSelectedBarNumber ?? 1);
      const beatLabel = eventGridLocation(item);
      editRange.textContent = `${beatLabel} · ${preciseTime(item.start)}–${preciseTime(item.end)}`;
      chordInput.value = item.chord_display || item.chord || "N";
      editStatus.textContent = `${editedEventCount()} 修改`;
      const selectedRow = document.querySelector(`.event[data-event="${selectedEventIndex}"]`);
      if (selectedRow) selectedRow.classList.add("selected");
      const selectedBar = document.querySelector(`.bar[data-bar="${item.bar}"]`);
      if (selectedBar) selectedBar.classList.add("selected");
      document.querySelectorAll(`.span-cell[data-event="${selectedEventIndex}"]`).forEach((node) => node.classList.add("selected"));
      updateSelectionDetails(item);
    }

    function selectEvent(index) {
      if (!events.length) return;
      selectedBeatTarget = null;
      selectedEventIndex = clamp(Number(index) || 0, 0, events.length - 1);
      const item = events[selectedEventIndex];
      if (item) lastSelectedBarNumber = Number(item.bar ?? lastSelectedBarNumber ?? 1);
      if (item && scoreStart) scoreStart.value = String(item.bar ?? 1);
      updateEditPanel();
    }

    function selectEventByObject(item) {
      const index = eventIndexByIdentity.get(item) ?? -1;
      if (index >= 0) selectEvent(index);
    }

    function selectBeat(barNumber, beatNumber) {
      const beat = beatByNumber(barNumber, beatNumber);
      if (!beat) return;
      selectedBeatTarget = { bar: Number(barNumber), beat: Number(beatNumber) };
      lastSelectedBarNumber = Number(barNumber);
      const eventIndex = events.findIndex((event) => beatCenter(beat) >= event.start && beatCenter(beat) < event.end);
      if (eventIndex >= 0) selectedEventIndex = eventIndex;
      if (scoreStart) scoreStart.value = String(barNumber || 1);
      updateEditPanel();
    }

    function saveSelectedBeat(parsed) {
      if (!canEditTimeline()) return false;
      if (!selectedBeatTarget) return false;
      const selection = { ...selectedBeatTarget };
      const bar = barByNumber(selection.bar);
      const beat = beatByNumber(selection.bar, selection.beat);
      if (!bar || !beat) return false;
      replaceTimeRangeInBar(bar, Number(beat.start), Number(beat.end), parsed, true);
      refreshDerived();
      renderStable(); selectBeat(selection.bar, selection.beat); updateActive(); autosaveDraft("beat");
      return true;
    }

    function saveSelectedChord() {
      if (!canEditTimeline()) return;
      if (!events.length) return;
      const parsed = parseEditableChord(chordInput.value);
      if (!parsed) {
        chordInput.setAttribute("aria-invalid", "true");
        toast("未识别的和弦写法，请输入例如 Am7、C/E、V7 或 N。");
        chordInput.focus();
        chordInput.select();
        return;
      }
      if (selectedBeatTarget && saveSelectedBeat(parsed)) {
        return;
      }
      const item = events[selectedEventIndex];
      updateCoveredItems(item, parsed, 1);
      dirtyEvents.add(selectedEventIndex);
      refreshDerived();
      renderStable();
      selectEvent(selectedEventIndex);
      updateActive();
      autosaveDraft("chord");
    }

    function selectedBarNumber(preferLastSelected = false) {
      if (selectedBeatTarget) return Number(selectedBeatTarget.bar);
      if (preferLastSelected && lastSelectedBarNumber) return Number(lastSelectedBarNumber);
      const item = events[selectedEventIndex];
      if (item) return Number(item.bar);
      return Number(lastSelectedBarNumber || (DATA.bars || [])[0]?.bar || 1);
    }

    function selectedCellItem() {
      if (selectedBeatTarget) {
        return beatByNumber(selectedBeatTarget.bar, selectedBeatTarget.beat);
      }
      return events[selectedEventIndex] || null;
    }

    function parsedFromItem(item) {
      return {
        chord: item.chord || "N",
        chordDisplay: item.chord_display || item.chord || "N",
        roman: item.roman || "N",
        romanDisplay: item.roman_display || item.roman || "N",
        bassRoot: item.bass_root || "N",
        bassConfidence: Number(item.bass_confidence || 0),
      };
    }

    function setCopyStatus(text) {
      copyStatus.textContent = text;
    }

    function copySelectedCell() {
      const item = selectedCellItem();
      if (!item) {
        setCopyStatus("没有可复制的单元格");
        return;
      }
      editClipboard = {
        type: "cell",
        parsed: cloneData(parsedFromItem(item)),
        confidence: Number(item.confidence || 1),
        label: labelFor(item),
      };
      setCopyStatus(`已复制格 ${editClipboard.label}`);
    }

    function pasteSelectedCell() {
      if (!canEditTimeline()) return;
      if (!editClipboard || editClipboard.type !== "cell") {
        setCopyStatus("没有单元格可粘贴");
        return;
      }
      const parsed = parseEditableChord(editClipboard.parsed.chordDisplay || editClipboard.parsed.chord) || cloneData(editClipboard.parsed);
      if (selectedBeatTarget && saveSelectedBeat(parsed)) {
        setCopyStatus(`已粘贴格 ${editClipboard.label}`);
        return;
      }
      const item = events[selectedEventIndex];
      if (!item) {
        setCopyStatus("没有可粘贴的位置");
        return;
      }
      updateCoveredItems(item, parsed, Number(editClipboard.confidence || 1), true);
      dirtyEvents.add(selectedEventIndex);
      refreshDerived();
      renderStable();
      selectEvent(selectedEventIndex);
      updateActive();
      setCopyStatus(`已粘贴格 ${editClipboard.label}`);
      autosaveDraft("paste-cell");
    }

    function copySelectedBar() {
      const bar = barByNumber(selectedBarNumber(true));
      const sourceBeats = bar ? [...(bar.beats || [])].sort((left, right) => left.beat - right.beat) : [];
      const sourceSpans = bar ? [...(bar.spans || [])].sort((left, right) => Number(left.start) - Number(right.start)) : [];
      if (!bar || !sourceBeats.length) {
        setCopyStatus("没有可复制的小节");
        return;
      }
      const duration = barDuration(bar);
      editClipboard = {
        type: "bar",
        sourceBar: Number(bar.bar),
        barStart: Number(bar.start || 0),
        barDuration: duration,
        beats: sourceBeats.map((beat) => ({
          parsed: cloneData(parsedFromItem(beat)),
          confidence: Number(beat.confidence || 1),
          label: labelFor(beat),
        })),
        spans: sourceSpans.map((span) => ({
          startRatio: clamp((Number(span.start || bar.start) - Number(bar.start || 0)) / duration, 0, 1),
          endRatio: clamp((Number(span.end || bar.end) - Number(bar.start || 0)) / duration, 0, 1),
          parsed: cloneData(parsedFromItem(span)),
          confidence: Number(span.confidence || 1),
          label: labelFor(span),
        })),
      };
      lastSelectedBarNumber = Number(bar.bar);
      setCopyStatus(`已复制小节 ${String(bar.bar).padStart(3, "0")}`);
    }

    function pastedSpanEventsForBar(bar) {
      if (!editClipboard || editClipboard.type !== "bar" || !editClipboard.spans?.length) return [];
      const start = Number(bar.start || 0);
      const duration = barDuration(bar);
      return editClipboard.spans
        .map((span) => {
          const left = start + clamp(Number(span.startRatio || 0), 0, 1) * duration;
          const right = start + clamp(Number(span.endRatio || 1), 0, 1) * duration;
          if (right <= left + 0.01) return null;
          const parsed = withKeyAtTime(left, () => parseEditableChord(span.parsed.chordDisplay || span.parsed.chord)) || cloneData(span.parsed);
          return eventFromTimeRange(bar, left, right, parsed, Number(span.confidence ?? 1), true);
        })
        .filter(Boolean);
    }

    function pasteSelectedBar() {
      if (!canEditTimeline()) return;
      if (!editClipboard || editClipboard.type !== "bar" || (!editClipboard.beats?.length && !editClipboard.spans?.length)) {
        setCopyStatus("没有小节可粘贴");
        return;
      }
      const targetBarNumber = selectedBarNumber(true);
      const bar = barByNumber(targetBarNumber);
      const targetBeats = bar ? [...(bar.beats || [])].sort((left, right) => left.beat - right.beat) : [];
      if (!bar || !targetBeats.length) {
        setCopyStatus(`没有可粘贴的小节：${String(targetBarNumber || "?").padStart(3, "0")}`);
        return;
      }
      lastSelectedBarNumber = Number(bar.bar);
      if (editClipboard.beats?.length) {
        for (let index = 0; index < targetBeats.length; index += 1) {
          const sourceIndex = Math.min(
            editClipboard.beats.length - 1,
            Math.floor((index * editClipboard.beats.length) / targetBeats.length)
          );
          const source = editClipboard.beats[sourceIndex];
          const parsed = withKeyAtTime(targetBeats[index].start, () => parseEditableChord(source.parsed.chordDisplay || source.parsed.chord)) || cloneData(source.parsed);
          applyChordFields(targetBeats[index], parsed, Number(source.confidence ?? 1), true);
        }
      }
      const pastedSpans = pastedSpanEventsForBar(bar);
      if (pastedSpans.length) replaceBarSpans(bar, pastedSpans);
      else rebuildBarSpansFromBeats(bar);
      refreshDerived();
      const firstEventIndex = events.findIndex((event) => Number(event.bar) === Number(bar.bar));
      selectedEventIndex = firstEventIndex >= 0 ? firstEventIndex : selectedEventIndex;
      const keepBeatSelection = selectedBeatTarget && Number(selectedBeatTarget.bar) === Number(bar.bar);
      const beatSelection = keepBeatSelection ? { ...selectedBeatTarget } : null;
      renderStable();
      if (beatSelection) selectBeat(beatSelection.bar, beatSelection.beat);
      else selectEvent(selectedEventIndex);
      updateActive();
      setCopyStatus(`已粘贴小节 ${String(editClipboard.sourceBar).padStart(3, "0")} -> ${String(bar.bar).padStart(3, "0")}`);
      autosaveDraft("paste-bar");
    }

    function restoreSelectedChord() {
      if (!canEditTimeline()) return;
      const selected = selectedCellItem();
      if (!selected) return;
      const range = { start:Number(selected.start),end:Number(selected.end) };
      let restoredAny = false;
      for (const bar of (DATA.bars || []).filter((candidate) => overlaps(candidate,range))) {
        const sourceBar = (SOURCE_DATA.bars || []).find((candidate) => Number(candidate.bar) === Number(bar.bar));
        const local = { start:Math.max(range.start,Number(bar.start)),end:Math.min(range.end,Number(bar.end)) };
        const originals = (sourceBar?.spans || SOURCE_DATA.events || []).filter((span) => overlaps(span,local));
        if (!originals.length) continue;
        const pieces = [];
        for (const span of bar.spans || []) {
          if (!overlaps(span,local)) { pieces.push({ ...span }); continue; }
          if (Number(span.start) < local.start-.0001) pieces.push({ ...span,end:local.start });
          if (Number(span.end) > local.end+.0001) pieces.push({ ...span,start:local.end });
        }
        for (const original of originals) {
          const left = Math.max(local.start,Number(original.start));
          const right = Math.min(local.end,Number(original.end));
          if (right-left < .0001) continue;
          const restored = { ...original,bar:Number(bar.bar),start:left,end:right };
          delete restored.edited;
          restored.beat_start = bar.beats?.length ? Number(beatPositionAtTime(bar,left).toFixed(3)) : null;
          restored.beat_end = bar.beats?.length ? Number(beatPositionAtTime(bar,right-.0001).toFixed(3)) : null;
          pieces.push(restored);
        }
        replaceBarSpans(bar,pieces);
        restoredAny = true;
      }
      if (!restoredAny) { toast("原始分析中没有对应片段。"); return; }
      dirtyEvents.clear();
      selectedEventIndex = Math.max(0,events.findIndex((event) => Number(event.start) >= range.start-.0001));
      renderStable(); updateActive(); autosaveDraft("restore-chord");
    }

    function resetAllEdits() {
      const fresh = cloneData(SOURCE_DATA);
      for (const key of Object.keys(DATA)) delete DATA[key];
      Object.assign(DATA, fresh);
      dirtyEvents.clear();
      selectedBeatTarget = null;
      refreshDerived();
      renderStable();
      selectEvent(0);
      updateActive();
      clearAutosaveDraft();
      if (editStatus) editStatus.textContent = "已重置，自动保存已清除";
      recordHistory("reset");
    }

    function exportEventsAsLab() {
      const lines = events.map((event) => {
        const label = event.chord_display || event.chord || "N";
        return `${Number(event.start).toFixed(3)} ${Number(event.end).toFixed(3)} ${label}`;
      });
      downloadText(`${baseExportName()}_corrected.lab`, `${lines.join("\n")}\n`, "text/plain;charset=utf-8");
    }

    function exportAnalysisJson() {
      const output = cloneData(DATA);
      output.corrected = editedEventCount() > 0 || Boolean(DATA.structure?.manual_overrides?.length) || Boolean(DATA.timeline_reconciliation);
      output.corrected_event_count = editedEventCount();
      output.corrected_at = new Date().toISOString();
      downloadText(`${baseExportName()}_corrected_analysis.json`, JSON.stringify(output, null, 2), "application/json;charset=utf-8");
      autosaveDraft("export-json");
    }

    function baseExportName() {
      const title = DATA.display_title || String(DATA.audio_file || "song").replace(/\.[^.]+$/, "");
      return String(title).replace(/[\\/:*?"<>|]+/g, "_");
    }

    function downloadText(filename, text, mimeType) {
      const blob = new Blob([text], { type: mimeType });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }

    function labelFor(item) {
      return displayMode === "roman"
        ? (item.roman_display || item.roman || item.chord_display || item.chord)
        : (item.chord_display || item.chord);
    }

    function suggestionKey(item) {
      return [Number(item.bar || 0), Number(item.start || 0).toFixed(3), Number(item.end || 0).toFixed(3), item.chord || item.to || item.suggested_chord || ""].join("|");
    }

    function formatSuggestionScore(value) {
      const score = Number(value || 0);
      return `证据 ${Number.isFinite(score) ? score.toFixed(2) : "未知"}`;
    }

    function selectedBarContext() {
      if (selectedBeatTarget) return Number(selectedBeatTarget.bar || 0);
      const item = events[selectedEventIndex];
      return Number(item?.bar || lastSelectedBarNumber || 0);
    }

    function resolveSuggestionLabel(item) {
      return item.suggested_chord || item.to || item.chord || item.label || item.name || "N";
    }

    function resolveProposalRange(proposal) {
      const cells = Array.isArray(proposal.cells) ? proposal.cells : [];
      if (!cells.length) return null;
      const starts = cells.map((cell) => Number(cell.start || 0));
      const ends = cells.map((cell) => Number(cell.end || 0));
      return { start: Math.min(...starts), end: Math.max(...ends) };
    }

    function renderSuggestionPanel() {
      if (!suggestionList) return;
      const currentBar = selectedBarContext();
      const suggestions = [...(DATA.extension_suggestions || [])].sort((left, right) => {
        const leftBar = Number(left.bar || 0);
        const rightBar = Number(right.bar || 0);
        if (leftBar !== rightBar) return leftBar - rightBar;
        return Number(right.score || right.suggestion_score || 0) - Number(left.score || left.suggestion_score || 0);
      });
      const visible = suggestions.filter((item) => !currentBar || Number(item.bar || 0) === Number(currentBar));
      const rows = visible.slice(0, 12);
      if (!rows.length) {
        suggestionList.innerHTML = '<div class="hint-empty">当前小节暂无候选建议。</div>';
        return;
      }
      suggestionList.innerHTML = rows.map((item) => {
        const bar = Number(item.bar || 0);
        const start = Number(item.start || 0);
        const end = Number(item.end || start);
        const label = resolveSuggestionLabel(item);
        const roman = item.roman ? ` · ${item.roman}` : "";
        const eligibility = item.eligible ? '<span class="hint-tag">可复核候选</span>' : '<span class="hint-tag warn">证据不足</span>';
        const tags = [item.kind, item.family, item.source].filter(Boolean).map((tag) => `<span class="hint-tag">${escapedAttr(tag)}</span>`).join("");
        const evidence = [
          item.flat7_over_triad != null ? `b7 ${Number(item.flat7_over_triad).toFixed(3)}` : null,
          item.maj7_over_triad != null ? `maj7 ${Number(item.maj7_over_triad).toFixed(3)}` : null,
          item.ninth_over_triad != null ? `9 ${Number(item.ninth_over_triad).toFixed(3)}` : null,
          item.score != null ? `score ${Number(item.score).toFixed(3)}` : null,
        ].filter(Boolean).join(" · ");
        return `
          <div class="hint-item" role="button" tabindex="0" data-bar="${bar}" data-start="${start}" data-end="${end}" data-key="${escapedAttr(suggestionKey(item))}">
            <strong>${escapedAttr(label)}${escapedAttr(roman)}</strong>
            <div class="meta">${String(bar).padStart(3, "0")} · ${fmt(start)}-${fmt(end)} · ${formatSuggestionScore(item.suggestion_score ?? item.score)}${item.eligible ? "" : " · 需复核"}</div>
            <div class="tag-row">${eligibility}${tags}</div>
            <div class="detail">${escapedAttr(evidence || (item.note || item.reason || ""))}</div>
            <button class="tool-button small" type="button" data-candidate="${escapedAttr(label)}">填入编辑器</button>
          </div>
        `;
      }).join("");
    }

    function renderPathProposalPanel() {
      if (!pathProposalList) return;
      const currentBar = selectedBarContext();
      const proposals = [...(DATA.path_proposals || [])].sort((left, right) => Number(right.score || 0) - Number(left.score || 0));
      const visible = proposals.filter((item) => !currentBar || Number(item.bar || 0) === Number(currentBar));
      const rows = visible.slice(0, 8);
      if (!rows.length) {
        pathProposalList.innerHTML = '<div class="hint-empty">当前分析里没有切分路径候选。</div>';
        return;
      }
      pathProposalList.innerHTML = rows.map((item) => {
        const range = resolveProposalRange(item) || { start: Number(item.start || 0), end: Number(item.end || 0) };
        const labels = [
          item.unit ? `<span class="hint-tag">${escapedAttr(item.unit)}</span>` : "",
          item.chosen ? '<span class="hint-tag warn">当前路径</span>' : "",
        ].join("");
        const summary = [
          item.bar_score != null ? `bar ${Number(item.bar_score).toFixed(3)}` : null,
          item.avg_top_score != null ? `top ${Number(item.avg_top_score).toFixed(3)}` : null,
          item.avg_margin != null ? `margin ${Number(item.avg_margin).toFixed(3)}` : null,
          item.strong_root_count != null ? `strong ${Number(item.strong_root_count)}` : null,
          item.root_switches != null ? `switch ${Number(item.root_switches)}` : null,
          item.distinct_roots != null ? `roots ${Number(item.distinct_roots)}` : null,
        ].filter(Boolean).join(" · ");
        return `
          <div class="hint-item" role="button" tabindex="0" data-bar="${Number(item.bar || 0)}" data-start="${Number(range.start || 0)}" data-end="${Number(range.end || 0)}" data-key="${escapedAttr(item.unit || "")}">
            <strong>Bar ${String(Number(item.bar || 0)).padStart(3, "0")}</strong>
            <div class="meta">${String(Number(item.bar || 0)).padStart(3, "0")} · ${escapedAttr(item.unit || "auto")} · ${formatSuggestionScore(item.score)}</div>
            <div class="tag-row">${labels}</div>
            <div class="detail">${escapedAttr(summary)}</div>
          </div>
        `;
      }).join("");
    }

    function labelWithBadge(item) {
      const label = escapedAttr(labelFor(item));
      const badge = item.extension_refined ? `<span class="badge" title="后验扩展修正">E</span>` : "";
      return `${label}${badge}`;
    }

    function titleFor(item) {
      const name = item.chord_display || item.chord;
      const romanName = item.roman_display || item.roman;
      const roman = romanName && romanName !== name ? ` / ${romanName}` : "";
      const bass = item.bass_root && item.bass_root !== "N" ? ` · 低音 ${item.bass_root} · 证据 ${Number(item.bass_confidence || 0).toFixed(2)}` : "";
      const refinement = item.extension_refined && item.extension_refinement
        ? ` · 后验扩展 ${item.extension_refinement.from || "?"}->${item.extension_refinement.to || name}`
          + ` · b7 ${Number(item.extension_refinement.flat7_over_triad || 0).toFixed(3)}`
          + ` · maj7 ${Number(item.extension_refinement.maj7_over_triad || 0).toFixed(3)}`
          + ` · 9 ${Number(item.extension_refinement.ninth_over_triad || 0).toFixed(3)}`
        : "";
      return `${name}${roman} · ${evidenceLabel(item)}${bass}${refinement} · ${evidenceExplanation(item)}`;
    }

    function escapedAttr(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/"/g, "&quot;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function spanPositionText(bar, span) {
      const start = sixteenthLabel(bar, Number(span.start || bar.start));
      const end = sixteenthLabel(bar, Math.max(Number(span.start || bar.start), Number(span.end || bar.end) - 0.001));
      return start === end ? start : `${start}-${end}`;
    }

    function renderSpanLane(bar) {
      const spans = [...(bar.spans || [])].sort((left, right) => Number(left.start) - Number(right.start));
      const cells = spans.map((span) => {
        const left = barFraction(bar, Number(span.start || bar.start)) * 100;
        const right = barFraction(bar, Number(span.end || bar.end)) * 100;
        const width = Math.max(0.1, right - left);
        const exactIndex = eventIndexBySpan.get(spanIdentity(span));
        const covering = activeFor((Number(span.start)+Number(span.end))/2);
        const eventIndex = exactIndex ?? eventIndexByIdentity.get(covering) ?? -1;
        const activeClass = eventIndex === selectedEventIndex && !selectedBeatTarget ? " selected" : "";
        const position = bar.grid_status === "uncovered" ? (bar.spans.length > 8 ? `${preciseTime(span.start)}–${preciseTime(span.end)}` : "拍点未定位") : bar.grid_status === "incomplete" ? `${preciseTime(span.start)}–${preciseTime(span.end)}` : span.position || spanPositionText(bar, span);
        return `
          <div class="span-cell${span.edited ? " edited" : ""}${span.extension_refined ? " extension-refined" : ""}${activeClass}" data-event="${eventIndex}" data-bar="${bar.bar}" data-start="${span.start}" data-end="${span.end}" style="left:${left.toFixed(3)}%;width:${width.toFixed(3)}%;" title="${escapedAttr(titleFor(span))} · ${escapedAttr(position)}">
            <span>${labelWithBadge(span)}</span>
            <span class="pos">${escapedAttr(position)}</span>
          </div>
        `;
      }).join("");
      return `
        <div class="span-lane" style="--subdivisions:${Math.max(1,beatCountForBar(bar) * 16 / meterDenominatorForBar(bar))}">
          ${["uncovered","incomplete"].includes(bar.grid_status) ? "" : '<div class="span-grid"></div>'}
          ${cells}
        </div>
      `;
    }

    function activeIn(items, time) {
      let lo = 0;
      let hi = items.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const item = items[mid];
        if (time < item.start) hi = mid - 1;
        else if (time >= item.end) lo = mid + 1;
        else return item;
      }
      return null;
    }

    function activeFor(time) {
      return activeIn(events, time);
    }

    function beatFor(time) {
      return activeIn(beats, time);
    }

    function render() {
      document.getElementById("songTitle").textContent = DATA.display_title || String(DATA.audio_file || "未命名音频").replace(/\.[^.]+$/, "");
      document.getElementById("songPath").textContent = `${gridCountLabel(DATA.bars)} · ${fmt(playbackDuration)} · 本地标注工作区`;
      document.getElementById("songPath").title = DATA.source || "";
      document.getElementById("bpm").textContent = bpmLabel();
      document.getElementById("key").textContent = `${DATA.key} ${DATA.mode}`;
      renderKeyFusionEvidence();
      document.getElementById("meter").textContent = DATA.meter;
      document.getElementById("barCount").textContent = DATA.bars.filter((bar) => bar.grid_status !== "uncovered").length;
      durationEl.textContent = fmt(playbackDuration);
      const nextAudioSrc = localAudioUrl || sessionAudioSrc;
      if (audio.getAttribute("src") !== nextAudioSrc) {
        audio.src = nextAudioSrc;
      }
      displaySync();
      displayButtons.forEach((button) => {
        button.classList.toggle("active", button.dataset.display === displayMode);
        button.setAttribute("aria-pressed", String(button.dataset.display === displayMode));
      });

      barsEl.style.setProperty("--beats", DATA.beats_per_bar);
      barsEl.innerHTML = DATA.bars.map((bar) => {
        const modulation = modulationForBar(bar.bar);
        const modulationLabel = modulation && modulation.index === 0 ? "主体调" : "转调";
        const modulationMarker = modulation ? `
          <div class="modulation-marker">${modulationLabel}：${escapedAttr(modulation.key)} ${escapedAttr(modulation.mode || "")} · ${escapedAttr(modulation.progression || "")}</div>
        ` : "";
        const beats = bar.beats.map((beat) => `
          <div class="beat${beat.edited ? " edited" : ""}${beat.extension_refined ? " extension-refined" : ""}" data-bar="${bar.bar}" data-beat="${beat.beat}" data-start="${beat.start}" style="grid-column:${Number(beat.beat)}" title="${escapedAttr(titleFor(beat))}">
            ${labelWithBadge(beat)}
          </div>
        `).join("");
        return `
          ${sectionHeadingForBar(bar)}
          <article class="bar${modulation ? " modulation" : ""}${barNeedsReview(bar) ? " low-confidence" : ""}${barMissingEvidence(bar) ? " missing-confidence" : ""}${bar.grid_status === "uncovered" ? ` uncovered${bar.spans.length > 8 ? ' uncovered-long' : ''}` : ""}" data-bar="${bar.bar}" data-start="${bar.start}" style="--beats:${Math.max(1,beatCountForBar(bar))}" tabindex="0" aria-label="${bar.grid_status === "uncovered" ? "拍点未定位区间" : `第 ${bar.bar} 小节`}，${escapedAttr((bar.spans || []).map(labelFor).join('，'))}">
            <div class="bar-head"><span>${bar.grid_status === "uncovered" ? "拍点未定位 · 时间片段" : String(bar.bar).padStart(3, "0")}${bar.grid_status === "incomplete" ? " · 拍点待核对" : bar.meter && String(bar.meter) !== String(DATA.meter) ? ` · ${escapedAttr(bar.meter)}` : ""}</span><span>${fmt(bar.start)}-${fmt(bar.end)}</span></div>
            ${modulationMarker}
            ${renderSpanLane(bar)}
            <div class="beats">${beats}</div>
            <div class="bar-playhead" aria-hidden="true"></div>
          </article>
        `;
      }).join("");

      eventList.innerHTML = events.map((event, index) => {
        const beat = eventGridLocation(event);
        return `
          <div class="event${event.edited ? " edited" : ""}${event.extension_refined ? " extension-refined" : ""}" role="button" tabindex="0" data-event="${index}" data-start="${event.start}" title="${escapedAttr(titleFor(event))}">
            <span class="when">${escapedAttr(beat)}</span>
            <strong>${labelWithBadge(event)}</strong>
            <span class="conf" title="${escapedAttr(evidenceExplanation(event))}">${event.edited ? "已修改" : confidenceScore(event) == null ? "未提供" : confidenceScore(event).toFixed(2)}</span>
          </div>
        `;
      }).join("");
      renderSuggestionPanel();
      renderPathProposalPanel();
      updateEditPanel();
      renderWorkbench();
    }

    function renderKeyFusionEvidence() {
      const panel = document.getElementById("keyFusionEvidence"), fusion = DATA.key_fusion;
      if (!panel) return;
      panel.replaceChildren(); panel.hidden = !fusion;
      if (!fusion) return;
      const add = (tag, text) => { const node = document.createElement(tag); node.textContent = text; panel.append(node); return node; };
      add("h3", `实验主调 ${fusion.key || "未定"} · ${fusion.needs_review ? "需审核" : "自动融合"}${(fusion.conflict?.primary_disagreement ?? fusion.conflict?.detected) ? " · 证据冲突" : ""}`);
      const names = {madmom_cnn:"CNN 原判",raw_chord_functional:"主证据：全曲和声",functional_joint:"辅助：音频与和声联合判断"};
      for (const [id,name] of Object.entries(names)) {
        const evidence = fusion.evidence?.[id];
        add("p", `${name}：${evidence?.key || "缺失"}；候选 ${(evidence?.candidates || []).slice(0,3).map(r => `${r.key}（排名 ${r.rank}，分数 ${Number(r.score).toFixed(4)}）`).join("、")}`);
      }
      const timeline = fusion.evidence?.window_timeline;
      add("p", `稳定小节覆盖：${Object.entries(timeline?.bar_shares || {}).map(([key,share]) => `${key} ${(share*100).toFixed(1)}%`).join("、") || "不足"}；${timeline?.modulation_detected ? "存在局部调性变化线索，需听辨" : "未确认局部转调"}`);
      if (timeline?.stability_schema) {
        const pct = value => value == null ? "无足够证据" : (value * 100).toFixed(1) + "%";
        add("p", `可判定 ${timeline.decisive_bars}/${timeline.valid_bars}（${pct(timeline.decisive_coverage)}）；未判定 ${timeline.unresolved_bars}/${timeline.valid_bars}（${pct(timeline.unresolved_coverage)}）。未判定不等于反对某个调。`);
        for (const key of new Set([fusion.evidence?.raw_chord_functional?.key, fusion.evidence?.madmom_cnn?.key].filter(Boolean))) {
          const m = timeline.candidate_metrics?.[key]; if (!m) continue;
          add("p", `${key}：可判定部分中占 ${pct(m.candidate_decisive_share)}；全部有效小节中占 ${pct(m.stable_coverage)}。最大竞争调 ${m.top_competitor_key || "无"} 占全部 ${pct(m.top_competitor_share_valid)}。`);
        }
      }
      for (const segment of timeline?.segments || []) {
        const button = add("button", `${fmt(segment.start)}–${fmt(segment.end)} ${segment.key} · 局部候选`);
        button.type = "button"; button.onclick = () => { audio.currentTime = segment.start; };
      }
      const gateNames = {three_sources:"三路证据齐全",complete_candidates:"完整的 24 调候选",joint_unique:"独立和声有唯一首选",harmonic_sufficient:"和弦证据充足",harmonic_unique:"和声候选有足够分差",joint_harmonic_agree:"两路和声判断一致",stable_windows:"足够多小节持续支持",cnn_support:"CNN 候选也有支持"};
      const failed = Object.entries(fusion.gate_checks || {}).filter(([,passed]) => !passed).map(([name]) => gateNames[name] || name);
      const decisions = {agreement:"声学与全曲和声一致",disagreement_resolved_cnn:"分歧裁决保留声学主调",disagreement_resolved_harmonic:"分歧裁决采用和声主调",ambiguous:"分歧尚无法可靠裁决，请审核"};
      add("p", `${fusion.decision_path ? decisions[fusion.decision_path] : (failed.length ? "暂未形成共识，未满足：" + failed.join("、") : "多来源共识门槛通过")}。分数不是已校准的正确概率；局部候选不自动改写和弦，主调仍需结合听辨。`);
      const thresholds = fusion.policy?.disagreement_thresholds;
      if (thresholds && thresholds.cnn > 1 && thresholds.harmonic > 1) add("p", "当前校准配置尚未启用自动分歧裁决；两路主证据一致时仍可接受。");
    }

    function modulationForBar(barNumber) {
      const modulations = Array.isArray(DATA.modulations) && DATA.modulations.length ? DATA.modulations : Array.isArray(DATA.key_timeline?.segments) ? DATA.key_timeline.segments : [];
      const index = modulations.findIndex((item) => Number(item.start_bar || 0) === Number(barNumber));
      return index >= 0 ? { ...modulations[index], index } : null;
    }

    function bpmLabel() {
      const base = Number(DATA.bpm || 0);
      const segments = Array.isArray(DATA.tempo_segments) ? DATA.tempo_segments : [];
      if (!segments.length) return base.toFixed(1);
      const values = segments
        .flatMap((segment) => [Number(segment.min_bpm || segment.bpm || 0), Number(segment.max_bpm || segment.bpm || 0)])
        .filter((value) => Number.isFinite(value) && value > 0);
      if (!values.length) return base.toFixed(1);
      const low = Math.min(...values);
      const high = Math.max(...values);
      if (high - low < 1.0) return base.toFixed(1);
      return `${base.toFixed(1)} (${low.toFixed(1)}-${high.toFixed(1)})`;
    }

    function isTextEditingTarget(target) {
      if (!target || target === document.body) return false;
      return Boolean(target.closest("input, textarea, select") || target.isContentEditable);
    }

    function handleCopyPasteShortcut(event) {
      if (!(event.ctrlKey || event.metaKey) || event.altKey || isTextEditingTarget(event.target)) return;
      const key = String(event.key || "").toLowerCase();
      if (key === "c") {
        event.preventDefault();
        if (event.shiftKey || !selectedBeatTarget) copySelectedBar();
        else copySelectedCell();
      } else if (key === "v") {
        event.preventDefault();
        if (event.shiftKey || editClipboard?.type === "bar") pasteSelectedBar();
        else if (editClipboard?.type === "cell") pasteSelectedCell();
        else if (!selectedBeatTarget) pasteSelectedBar();
        else pasteSelectedCell();
      }
    }

    function sortedBeats() {
      const allowed = new Set(navigableBars().map((bar) => Number(bar.bar)));
      return beats.filter((beat) => allowed.has(Number(beat.bar))).sort((left, right) => Number(left.bar) - Number(right.bar) || Number(left.beat) - Number(right.beat));
    }

    function navigableBars() {
      return DATA.bars.filter((bar) => !byId("reviewOnly").checked || barNeedsReview(bar));
    }

    function adjacentNavigableBar(barNumber, delta) {
      const ordered = navigableBars();
      return delta > 0 ? ordered.find((bar) => Number(bar.bar) > Number(barNumber)) : ordered.slice().reverse().find((bar) => Number(bar.bar) < Number(barNumber));
    }

    function moveEventSelection(delta) {
      const allowed = new Set(navigableBars().map((bar) => Number(bar.bar)));
      const ordered = events.map((event, index) => ({ event, index })).filter(({event}) => allowed.has(Number(event.bar)));
      if (!ordered.length) return false;
      const current = ordered.findIndex(({index}) => index === selectedEventIndex);
      const next = ordered[clamp(current + delta, 0, ordered.length - 1)];
      selectEvent(next.index);
      return true;
    }

    function moveBeatSelection(delta) {
      const ordered = sortedBeats();
      if (!ordered.length) return false;
      const currentIndex = selectedBeatTarget
        ? ordered.findIndex((beat) => Number(beat.bar) === Number(selectedBeatTarget.bar) && Number(beat.beat) === Number(selectedBeatTarget.beat))
        : -1;
      const nextIndex = clamp(currentIndex + delta, 0, ordered.length - 1);
      const next = ordered[nextIndex >= 0 ? nextIndex : 0];
      if (!next) return false;
      selectBeat(next.bar, next.beat);
      return true;
    }

    function moveBeatSelectionByBar(deltaBar) {
      if (!selectedBeatTarget) return false;
      const targetBar = adjacentNavigableBar(selectedBeatTarget.bar, deltaBar);
      if (!targetBar) return false;
      const targetBeats = [...(targetBar.beats || [])].sort((left, right) => Number(left.beat) - Number(right.beat));
      if (!targetBeats.length) return false;
      const wantedBeat = Number(selectedBeatTarget.beat);
      const next = targetBeats.reduce((best, beat) => {
        if (!best) return beat;
        return Math.abs(Number(beat.beat) - wantedBeat) < Math.abs(Number(best.beat) - wantedBeat) ? beat : best;
      }, null);
      if (!next) return false;
      selectBeat(next.bar, next.beat);
      return true;
    }

    function moveEventSelectionByBar(deltaBar) {
      if (!events.length) return false;
      const item = events[selectedEventIndex] || events[0];
      const targetBarNumber = adjacentNavigableBar(item.bar ?? 1, deltaBar)?.bar;
      const candidates = events
        .map((event, index) => ({ event, index }))
        .filter((entry) => Number(entry.event.bar) === targetBarNumber);
      if (!candidates.length) return false;
      const wantedBeat = Number(item.beat_start || 1);
      const next = candidates.reduce((best, entry) => {
        if (!best) return entry;
        return Math.abs(Number(entry.event.beat_start || 1) - wantedBeat) < Math.abs(Number(best.event.beat_start || 1) - wantedBeat) ? entry : best;
      }, null);
      if (!next) return false;
      selectEvent(next.index);
      return true;
    }

    function handleNavigationShortcut(event) {
      if (event.ctrlKey || event.metaKey || event.altKey || isTextEditingTarget(event.target)) return;
      const key = event.key;
      let handled = false;
      if (selectedBeatTarget) {
        if (key === "ArrowLeft") handled = moveBeatSelection(-1);
        else if (key === "ArrowRight") handled = moveBeatSelection(1);
        else if (key === "ArrowUp") handled = moveBeatSelectionByBar(-1);
        else if (key === "ArrowDown") handled = moveBeatSelectionByBar(1);
      } else if (events.length) {
        if (key === "ArrowLeft") {
          handled = moveEventSelection(-1);
        } else if (key === "ArrowRight") {
          handled = moveEventSelection(1);
        } else if (key === "ArrowUp") {
          handled = moveEventSelectionByBar(-1);
        } else if (key === "ArrowDown") {
          handled = moveEventSelectionByBar(1);
        }
      }
      if (handled) {
        event.preventDefault();
        const bar = selectedBarNumber();
        revealBar(bar, false);
        cachedBarNodes.get(Number(bar))?.focus({ preventScroll: true });
      }
    }

    barsEl.addEventListener("click", (event) => {
      const span = event.target.closest(".span-cell");
      if (span) {
        lastSelectedBarNumber = Number(span.dataset.bar || lastSelectedBarNumber || 1);
        const index = Number(span.dataset.event || -1);
        if (index >= 0) selectEvent(index);
        seekToAnalysis(Number(span.dataset.start || 0));
        return;
      }
      const beat = event.target.closest(".beat");
      if (beat) {
        lastSelectedBarNumber = Number(beat.dataset.bar || lastSelectedBarNumber || 1);
        selectBeat(Number(beat.dataset.bar || 0), Number(beat.dataset.beat || 0));
        seekToAnalysis(Number(beat.dataset.start || 0));
        return;
      }
      const bar = event.target.closest(".bar");
      if (!bar) return;
      lastSelectedBarNumber = Number(bar.dataset.bar || lastSelectedBarNumber || 1);
      const item = activeFor(Number(bar.dataset.start || 0) + 0.001);
      if (item) selectEventByObject(item);
      seekToAnalysis(Number(bar.dataset.start || 0));
    });

    eventList.addEventListener("click", (event) => {
      const row = event.target.closest(".event");
      if (!row) return;
      selectEvent(Number(row.dataset.event || 0));
      seekToAnalysis(Number(row.dataset.start || 0));
    });

    function jumpToHintTarget(node) {
      if (!node) return;
      const bar = Number(node.dataset.bar || 0);
      const start = Number(node.dataset.start || 0);
      if (Number.isFinite(start)) {
        seekToAnalysis(start);
      }
      if (bar > 0) {
        const match = events.findIndex((event) => Number(event.bar) === bar && Number(event.start) <= start + 0.02 && Number(event.end) >= start - 0.02);
        if (match >= 0) selectEvent(match);
        else {
          const fallback = events.findIndex((event) => Number(event.bar) === bar);
          if (fallback >= 0) selectEvent(fallback);
        }
      }
      revealBar(bar);
    }

    suggestionList?.addEventListener("click", (event) => {
      const row = event.target.closest(".hint-item");
      if (!row) return;
      jumpToHintTarget(row);
      const candidate = event.target.closest("[data-candidate]");
      if (candidate) {
        chordInput.value = candidate.dataset.candidate;
        activateInspector("editPanel");
        chordInput.focus({ preventScroll: true });
        chordInput.select();
        toast("候选已填入。试听后点击「应用」确认修改。");
      }
    });

    pathProposalList?.addEventListener("click", (event) => {
      const row = event.target.closest(".hint-item");
      if (!row) return;
      jumpToHintTarget(row);
    });

    displayButtons.forEach((button) => {
      button.addEventListener("click", () => {
        displayMode = button.dataset.display || "name";
        renderStable();
        updateActive();
      });
    });

    saveChord.addEventListener("click", saveSelectedChord);
    clearChord.addEventListener("click", () => {
      chordInput.value = "N";
      saveSelectedChord();
    });
    resetChord.addEventListener("click", restoreSelectedChord);
    splitHalf.addEventListener("click", () => splitSelectedEvent("half"));
    splitBeats.addEventListener("click", () => splitSelectedEvent("beat"));
    split8.addEventListener("click", () => splitSelectedEvent("eighth"));
    split16.addEventListener("click", () => splitSelectedEvent("sixteenth"));
    mergeSame.addEventListener("click", mergeAdjacentSameChords);
    copyCellBtn.addEventListener("click", copySelectedCell);
    pasteCellBtn.addEventListener("click", pasteSelectedCell);
    copyBarBtn.addEventListener("click", copySelectedBar);
    pasteBarBtn.addEventListener("click", pasteSelectedBar);
    scoreFile.addEventListener("change", () => loadScoreFile(scoreFile.files && scoreFile.files[0]));
    applyScore.addEventListener("click", applyScoreToTimeline);
    clearScore.addEventListener("click", () => {
      scoreText.value = "";
      scoreStatus.textContent = "等待曲谱";
      scoreFile.value = "";
    });
    exportLab.addEventListener("click", exportEventsAsLab);
    exportJson.addEventListener("click", exportAnalysisJson);
    resetAll.addEventListener("click", resetAllEdits);
    chordInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        saveSelectedChord();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (document.querySelector("dialog[open]")) return;
      handleWorkbenchShortcut(event);
      if (!event.defaultPrevented) handleCopyPasteShortcut(event);
      if (!event.defaultPrevented) handleNavigationShortcut(event);
    });

    function clearActive() {
      if (activeBar) activeBar.classList.remove("active");
      if (activeBeat) activeBeat.classList.remove("active");
      if (activeEvent) activeEvent.classList.remove("active");
      activeBar = null;
      activeBeat = null;
      activeEvent = null;
    }

    function setActive(nextBar, nextBeat, nextEvent) {
      if (nextBar === activeBar && nextBeat === activeBeat && nextEvent === activeEvent) return;
      clearActive();
      activeBar = nextBar;
      activeBeat = nextBeat;
      activeEvent = nextEvent;
      if (activeBar) activeBar.classList.add("active");
      if (activeBeat) activeBeat.classList.add("active");
      if (activeEvent) activeEvent.classList.add("active");
    }

    function updateActive() {
      enforcePlaybackRange();

      const localTime = localPlaybackTime();
      const lookupTime = analysisTime();
      currentTimeEl.textContent = fmt(localTime);
      seek.value = playbackDuration > 0 ? String(Math.round((localTime / playbackDuration) * Number(seek.max))) : "0";
      const item = activeFor(lookupTime);
      currentChord.textContent = item ? labelFor(item) : "—";

      const beat = beatFor(lookupTime);
      const barNumber = beat ? beat.bar : item?.bar;
      const beatNumber = beat ? beat.beat : item?.beat_start;
      const nextBar = cachedBarNodes.get(Number(barNumber));
      const nextBeat = cachedBeatNodes.get(`${barNumber}.${beatNumber}`);
      const eventIndex = eventIndexByIdentity.get(item) ?? -1;
      const nextEvent = cachedEventNodes.get(eventIndex);
      setActive(nextBar, nextBeat, nextEvent);
      updateWorkbenchPlayback(lookupTime, item, beat, nextBar, eventIndex);
    }

    function startTicker() {
      if (rafId !== null) return;
      const tick = () => {
        updateActive();
        rafId = audio.paused ? null : requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    }

    function stopTicker() {
      if (rafId !== null) cancelAnimationFrame(rafId);
      rafId = null;
      updateActive();
    }

    playBtn.addEventListener("click", () => {
      if (audio.paused) {
        if (audio.currentTime < playbackStart || audio.currentTime >= playbackEnd) {
          audio.currentTime = playbackStart;
        }
        playAudio();
      } else {
        audio.pause();
      }
    });
    audio.addEventListener("error", () => showPlaybackError(audio.error));

    audio.addEventListener("play", () => {
      playBtn.textContent = "⏸";
      startTicker();
    });
    audio.addEventListener("pause", () => {
      playBtn.textContent = "▶";
      stopTicker();
    });
    audio.addEventListener("timeupdate", () => {
      if (audio.paused) updateActive();
    });
    audio.addEventListener("loadedmetadata", () => {
      if (audio.currentTime < playbackStart) {
        audio.currentTime = playbackStart;
      }
      durationEl.textContent = fmt(playbackDuration || audio.duration || DATA.duration);
      updateActive();
    });

    seek.addEventListener("input", () => {
      audio.currentTime = playbackStart + (Number(seek.value) / Number(seek.max)) * playbackDuration;
      updateActive();
    });

    sync.addEventListener("input", () => {
      syncMs = Number(sync.value || 0);
      try {
        localStorage.setItem(syncStorageKey, String(syncMs));
      } catch (error) {
      }
      displaySync();
      updateActive();
    });

    // Initialization lives after the workbench extension in the same inline script.
