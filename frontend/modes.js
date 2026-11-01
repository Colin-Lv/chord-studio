"use strict";
/* Two front-end modes.
   体验版 keeps only listening surfaces. 评测版 exposes the full audit tooling, including the
   bar editor. Switching modes reloads the page because the bar editor rewires the DOM when it
   mounts; the open project travels through ?project= so the reload lands on the same song. */
(function () {
  const KEY = "chord.mode";
  const params = new URLSearchParams(location.search);
  const initial = params.get("mode") === "audit" ? "audit"
    : params.get("mode") === "user" ? "user"
      : (localStorage.getItem(KEY) === "audit" ? "audit" : "user");

  function paint(mode) {
    document.documentElement.dataset.mode = mode;
    for (const button of document.querySelectorAll("#modeSwitch button")) {
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    }
    const nav = document.getElementById("auditNav");
    if (nav) {
      nav.hidden = mode !== "audit";
      if (mode === "audit") fillNav(nav);
    }
  }

  function fillNav(nav) {
    const project = window.state && state.project;
    const id = project && project.id;
    for (const link of nav.querySelectorAll("[data-audit-page]")) {
      const page = link.dataset.auditPage;
      link.href = id ? `/${page}?project=${encodeURIComponent(id)}` : `/${page}`;
    }
  }

  /* The bar editor is an audit tool: in user mode it must never mount, because mounting moves
     the timeline and the chord table into its own collapsed details, and it also re-reads the
     playhead. Everything it offers is gated here instead of being hidden with CSS. */
  function gateEditor() {
    const editor = window.BarEditor;
    if (!editor || editor.__modeGated) return Boolean(editor);
    const open = editor.open;
    const playbackAt = editor.playbackAt;
    const toggleLoop = editor.toggleLoop;
    editor.open = function (...args) {
      if (document.documentElement.dataset.mode === "audit") return open.apply(editor, args);
      return undefined;
    };
    if (playbackAt) {
      editor.playbackAt = function (...args) {
        if (document.documentElement.dataset.mode === "audit") return playbackAt.apply(editor, args);
        return null;
      };
    }
    if (toggleLoop) {
      editor.toggleLoop = function (...args) {
        if (document.documentElement.dataset.mode === "audit") return toggleLoop.apply(editor, args);
        return undefined;
      };
    }
    editor.__modeGated = true;
    return true;
  }

  let tries = 0;
  const timer = setInterval(() => {
    tries += 1;
    if (gateEditor() || tries > 60) clearInterval(timer);
  }, 50);

  function watchResult() {
    const panel = document.getElementById("resultPanel");
    if (!panel) return;
    const sync = () => document.body.classList.toggle("has-result", !panel.hidden);
    new MutationObserver(sync).observe(panel, { attributes: true, attributeFilter: ["hidden"] });
    sync();
  }

  function switchTo(mode) {
    localStorage.setItem(KEY, mode);
    const url = new URL(location.href);
    const project = window.state && state.project && state.project.id;
    if (project) url.searchParams.set("project", project);
    url.searchParams.set("mode", mode);
    location.assign(url.toString());
  }

  function autoOpen() {
    const wanted = params.get("project") || sessionStorage.getItem("chord.open");
    if (!wanted) return;
    sessionStorage.setItem("chord.open", wanted);
    let attempts = 0;
    const tick = setInterval(() => {
      attempts += 1;
      const target = document.querySelector(`[data-project-id="${CSS.escape(wanted)}"]`);
      if (target) {
        clearInterval(tick);
        if (!target.classList.contains("active")) target.click();
      } else if (attempts > 60) {
        clearInterval(tick);
      }
    }, 200);
  }

  document.addEventListener("click", event => {
    const button = event.target.closest("#modeSwitch button");
    if (button && button.dataset.mode !== document.documentElement.dataset.mode) {
      switchTo(button.dataset.mode);
    }
  });

  document.addEventListener("keydown", event => {
    if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "m") {
      event.preventDefault();
      switchTo(document.documentElement.dataset.mode === "audit" ? "user" : "audit");
    }
  });

  document.addEventListener("chord:project", () => {
    const nav = document.getElementById("auditNav");
    if (nav && !nav.hidden) fillNav(nav);
  });

  window.chordMode = {
    current: () => document.documentElement.dataset.mode,
    switchTo,
  };

  paint(initial);
  watchResult();
  autoOpen();
})();
