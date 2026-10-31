// Headless check of the workbench renderer for the experimental key block.
// Extracts the real renderKeyExperimental() source from app.js, runs it against a minimal
// DOM stub and a real analysis.json payload, and asserts what the user should see.
const fs = require("fs");

const APP = String.raw`C:\Users\33573\AppData\Local\Programs\Chord Studio\backend\frontend\app.js`;
const ANALYSIS = process.argv[2];
const MODE = process.argv[3] || "";

function makeNode(tag, cls, text) {
  return {
    tag, className: cls || "", textContent: text === undefined ? "" : text,
    children: [], hidden: false, disabled: false, title: "", attrs: {},
    append(...nodes) { for (const node of nodes) this.children.push(node); },
    appendChild(node) { this.children.push(node); },
    replaceChildren() { this.children = []; },
    addEventListener(type, fn) { (this.listeners ||= {})[type] = fn; },
    setAttribute(name, value) { this.attrs[name] = value; },
  };
}

const textOf = node => !node ? "" : String(node.textContent || "") + (node.children || []).map(textOf).join(" ");
const all = (node, out = []) => { out.push(node); for (const c of node.children || []) all(c, out); return out; };

function keyIdentity(value) {
  const m = /^([A-G])([#b]?)(.*)$/.exec(String(value || "").trim());
  if (!m) return null;
  const pc = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1]] + ({ "#": 1, b: -1, "": 0 }[m[2]] || 0);
  const rest = m[3] || "";
  const minor = /^m(?!aj)/.test(rest) || /^:m(in(or)?)?$/.test(rest) || /minor$/i.test(rest);
  return ((pc % 12) + 12) % 12 + (minor ? "m" : "");
}

const source = fs.readFileSync(APP, "utf8");
const start = source.indexOf("function renderKeyExperimental(){");
if (start < 0) throw new Error("renderKeyExperimental not found in app.js");
const end = source.indexOf("\nfunction renderKey()", start);
if (end < 0) throw new Error("could not delimit renderKeyExperimental");
const body = source.slice(start, end).trim();

const analysis = JSON.parse(fs.readFileSync(ANALYSIS, "utf8"));
const block = analysis.key_analysis.experimental;
if (!block) throw new Error("analysis.json has no key_analysis.experimental");
const keyMode = MODE || analysis.key_mode || "madmom";
const displayedKey = keyMode === "continuous" && analysis.key_analysis.source === "continuous_key_v1"
  ? block.key : analysis.key_analysis.key;

const panel = makeNode("section", "panel key-experimental");
panel.hidden = true;
const adopted = [];
const keyName = value => value ? value.replace(/m$/, "") + (value.endsWith("m") ? " 小调" : " 大调") : "未确定";
const sandbox = {
  panel,
  element: makeNode,
  keyName,
  state: { result: { key_analysis: { experimental: block, source: analysis.key_analysis.source },
                     key_mode: keyMode },
           review: { revision: 0 }, key: displayedKey },
  $: id => (id === "keyExperimental" ? panel : null),
  updateKeyReview: (action, key) => adopted.push([action, key]),
};
new Function(...Object.keys(sandbox), body + "; return renderKeyExperimental();")(...Object.values(sandbox));

const nodes = all(panel);
const headings = nodes.filter(n => n.tag === "h3").map(n => n.textContent);
const buttons = nodes.filter(n => n.tag === "button");
const rows = nodes.filter(n => n.tag === "tr");
const table = nodes.find(n => n.tag === "table");
const text = textOf(panel);
const sameIdentity = keyIdentity(block.key) === keyIdentity(displayedKey);

const checks = {
  "panel visible": panel.hidden === false,
  "heading present": headings.some(v => v.includes("连续评分")),
  "heading reflects active mode": keyMode === "continuous"
    ? headings.some(v => v.includes("当前主调来源")) : headings.some(v => v.includes("第二档")),
  "table present": !!table && table.className.includes("key-experiment-table"),
  "header + candidate rows": rows.length === block.ranking.length + 1,
  "chosen row marked": rows.some(r => r.className === "chosen"),
  "summary mentions key": text.includes(keyName(block.key)),
  "summary mentions acoustic key": text.includes(keyName(block.acoustic_key)),
  "warnings rendered": (block.warnings || []).every(w => text.includes(w)),
  "known regressions rendered": (block.known_regressions || []).every(r => text.includes(r.case)),
  "adopt visibility uses enharmonic identity": (buttons[0] || {}).hidden === sameIdentity,
  "enharmonic keys treated as equal":
    keyIdentity("G#") === keyIdentity("Ab") && keyIdentity("C") !== keyIdentity("Cm"),
};

if (buttons[0] && buttons[0].listeners && buttons[0].listeners.click) {
  buttons[0].listeners.click();
  checks["adopt uses custom-key save"] = adopted.length === 1 && adopted[0][0] === "custom"
    && adopted[0][1] === block.key;
} else {
  checks["adopt uses custom-key save"] = sameIdentity;
}

let failed = 0;
for (const [name, ok] of Object.entries(checks)) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}`);
  if (!ok) failed += 1;
}
console.log("\nrendered text:\n" + text.replace(/\s+/g, " ").slice(0, 320));
process.exit(failed ? 1 : 0);
