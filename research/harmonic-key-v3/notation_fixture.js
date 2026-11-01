"use strict";window.HarmonyNotation=(()=>{let currentKey="";function editingKey(){return currentKey;}
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


return {setKey(value){currentKey=value;},parse:parseChord,parseInKey(label,key){const previous=currentKey;try{currentKey=key;return parseChord(label);}finally{currentKey=previous;}},parseEditable:parseEditableChord,roots:Object.keys(ROOT_TO_PC)};})();