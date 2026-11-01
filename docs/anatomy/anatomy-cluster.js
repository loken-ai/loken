(() => {
// The element that holds the plate, and the prefix of its ids.
const ROOT = document.querySelector(".ana-plate"), P = ROOT.dataset.prefix || "";
// The data of the plate, in the page next to its drawing.
const D = JSON.parse(ROOT.querySelector("script.ana-data").textContent);
const scoped = s => s.replace(/#(?=[\w-])/g, "#" + P);
const $ = s => ROOT.querySelector(scoped(s)), $$ = s => [...ROOT.querySelectorAll(scoped(s))];
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const color = g => `var(--a-g-${g})`;
const NS = "http://www.w3.org/2000/svg";
const plate = D.plate;
// Inline text of the data: HTML escaped, `code` spans kept.
const fmt = s => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>");

// The clock of the players.
// It stands still while paused.
// It runs at once while seeking a step.
const T = {
  paused: false, seeking: false, v: 0, at: performance.now(),
  now() {
    if (T.seeking) return (T.v += 1e6);
    if (!T.paused) { const n = performance.now(); T.v += n - T.at; T.at = n; }
    return T.v;
  },
  pause(on) {
    T.now(); T.paused = on; T.at = performance.now();
    ROOT.classList.toggle("paused", on);
    syncCtl();
  },
  wait(ms) {
    const due = T.now() + ms;
    return new Promise(r => {
      const check = () => {
        if (T.seeking) return r();
        const left = due - T.now();
        if (left <= 0) r(); else setTimeout(check, T.paused ? 100 : left);
      };
      check();
    });
  },
  raf(cb) {
    if (T.seeking) return queueMicrotask(() => cb(T.now()));
    requestAnimationFrame(function f() {
      if (T.paused && !T.seeking) return requestAnimationFrame(f);
      cb(T.now());
    });
  },
};

// The step controls: every call of a step counts, Next seeks the following one, Previous
// replays the scene up to the one before.
// `epoch` counts the stops: a timer of a stopped scene leaves the next one alone.
const CTL = {count: 0, target: 0, hold: false, key: null, want: null, epoch: 0};

// The step list of the scene being shown, in the text of the plate.
const stepItems = () => [...ROOT.querySelectorAll(`.script section[data-scene="${CTL.key}"] li`)];

function stepped() {
  CTL.count++;
  const lis = stepItems(), j = lis.findIndex(li => li.classList.contains("now"));
  lis.forEach((li, i) => i === j ? li.setAttribute("aria-current", "step") : li.removeAttribute("aria-current"));
  if (T.seeking && (CTL.want !== null ? j === CTL.want : CTL.count >= CTL.target)) {
    T.seeking = false; T.at = performance.now(); CTL.want = null;
    if (CTL.hold) T.pause(true);
    settle();
  }
  if (!T.seeking && j >= 0) $("#sr").textContent = `Step ${j + 1} of ${lis.length}: ${lis[j].textContent.replace(/\s+/g, " ").trim()}`;
  // The step box and the step bar change together, in the same call.
  if (j >= 0) BOX.step.innerHTML = `<b>${j + 1}</b> ${esc(lis[j].textContent.replace(/\s+/g, " ").trim())}`;
  syncCtl();
  layout();
}

function syncCtl() {
  const b = $("#play");
  if (!b) return;
  const idle = plate === "overview" ? "Play a request" : "Play";
  const label = !anim ? (CTL.count ? "Play again" : idle) : T.paused ? "Resume" : "Pause";
  b.textContent = label;
  b.setAttribute("aria-label", label === "Play again" ? "Play the scene again" : label === "Play" ? "Play the scene" : label);
  $("#prev").disabled = !anim || CTL.count < 2;
  const j = stepItems().findIndex(li => li.classList.contains("now"));
  $$(".dots button").forEach((d, i) => i === j ? d.setAttribute("aria-current", "step") : d.removeAttribute("aria-current"));
}

// Plays the scene `key` from the start, seeking step call `target` when it is above 0, or the
// step of index `want`, held there when `hold` is true.
function start(key, target, hold, want = null) {
  stopAnim();
  setScene(key);
  CTL.count = 0; CTL.target = target; CTL.hold = hold; CTL.want = want;
  T.seeking = target > 0 || want !== null;
  ROOT.classList.toggle("seeking", T.seeking);
  ROOT.dataset.state = "play";
  if (plate === "overview") play();
  else if (plate === "cluster") playCluster(key);
  else playScene(plate, key);
}

function next() {
  if (!anim) return start(CTL.key, 1, true);
  CTL.target = CTL.count + 1; CTL.hold = T.paused;
  T.seeking = true;
  ROOT.classList.add("seeking");
}

// The end of a seek: the styles reached are computed with no transition, then transitions
// come back for what happens next.
function settle() {
  void ROOT.offsetWidth;
  ROOT.classList.remove("seeking");
}

function prev() {
  if (!anim) return;
  start(CTL.key, Math.max(1, CTL.count - 1), T.paused);
}

// In a narrow column the plate drops the empty margins on its sides.
function crop() {
  const svg = $(`#v-${plate}`);
  if (!svg || svg.dataset.cropped) return;
  const b = svg.getBBox(), h = svg.viewBox.baseVal.height, pad = 12;
  if (!b.width) return;
  // A plate with no free zone in its drawing keeps columns beside it for the inlays.
  const [left, right] = (D.inlay && D.inlay[plate] && D.inlay[plate].pad) || [0, 0];
  const x0 = Math.max(0, b.x - pad) - left, w = Math.min(1000, b.width + 2 * pad) + left + right;
  svg.setAttribute("viewBox", `${x0} 0 ${w} ${h}`);
  svg.dataset.cropped = "1";
}

function show() {
  crop();
  $("#cap").textContent = D.plates[plate];
}

function nerve(from, to) {
  const g = $(`#v-${plate} .nerves`);
  const [ax, ay] = from, [lx, ly] = to;
  const cx = ax + (lx - ax) * .75 + (ax < 500 ? -18 : 18), cy = ay + (ly - ay) * .25;
  const p = document.createElementNS(NS, "path");
  p.setAttribute("class", "nerve");
  p.setAttribute("d", `M${ax},${ay} Q${cx},${cy} ${lx},${ly}`);
  const c = document.createElementNS(NS, "circle");
  c.setAttribute("class", "ndot"); c.setAttribute("cx", lx); c.setAttribute("cy", ly); c.setAttribute("r", 2.6);
  g.append(p, c);
}

function light(groups, sel, level) {
  $$("svg.plate").forEach(s => s.classList.toggle("has-sel", groups.length > 0 || !!level));
  $$("[data-group]").forEach(el => {
    if (!el.classList) return;
    el.classList.toggle("sel", el.dataset.group === sel);
    el.classList.toggle("lit", groups.includes(el.dataset.group));
  });
  $$("[data-level]").forEach(el => el.classList.toggle("sel", el.dataset.level === level));
  $$(".key button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.group === sel)));
  $$(".nerves").forEach(g => g.replaceChildren());
}

// What the inlays say of scene `key`: its title, the rows of its Now box, its closing note.
function meta(key) {
  if (plate === "overview")
    return {eyebrow: "Reading the map", title: "The way of a request", rows: ["Prompt", "Answer"],
            prompt: DEMO.prompt, note: [D.repeat], n: D.steps.length};
  if (plate === "cluster") {
    const m = D.modes[key];
    return {eyebrow: D.plates[plate], title: m.title, rows: ["Prompt", "Answer"], prompt: m.prompt || DEMO.prompt,
            note: m.notes || [], n: m.steps.length};
  }
  const s = sceneOf(plate, key);
  return {eyebrow: D.plates[plate], title: s.title, rows: s.rows || [], plan: s.plan, legend: s.legend,
          note: [s.repeat, ...(s.notes || []), ...(s.items || [])].filter(Boolean), n: s.steps.length};
}

// The example plan of a scene on the vertebrae, from layer 0 up: [class, count] runs.
const PLAN_CLASSES = ["pa", "pb", "ph", "pt"];
function applyPlan(plan) {
  const vts = [...$(`#v-${plate}`).querySelectorAll(".vt")].map(el => ({el, y: el.getBBox().y})).sort((a, b) => b.y - a.y);
  let i = 0;
  for (const [cls, n] of plan || []) for (let j = 0; j < n && i < vts.length; j++, i++) {
    vts[i].el.classList.remove(...PLAN_CLASSES);
    vts[i].el.classList.add(cls);
    vts[i].el.dataset.plan = cls;
  }
  for (; i < vts.length; i++) { vts[i].el.classList.remove(...PLAN_CLASSES); delete vts[i].el.dataset.plan; }
}

function setScene(key) {
  CTL.key = key;
  const m = meta(key);
  $$(".modes button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.scene === key)));
  $(".dots").innerHTML = [...Array(m.n)].map((_, i) =>
    `<button type="button" data-step="${i}" aria-label="Go to step ${i + 1}">${i + 1}</button>`).join("");
  BOX.title.innerHTML = `<span class="e">${esc(m.eyebrow)}</span><span class="t">${esc(m.title)}</span>`;
  BOX.now.innerHTML = `<span class="k">Now</span><span class="v lead" id="${P}now"></span>` + m.rows.map((r, i) =>
    r === "Prompt" ? `<span class="k">Prompt</span><span class="v text">${esc(m.prompt)}</span>`
    : r === "Answer" ? `<span class="k">Answer</span><span class="v" id="${P}answer"></span><span class="c" id="${P}count"></span>`
    : `<span class="k">${esc(r)}</span><span class="v" id="${P}row-${i}"></span>`).join("");
  BOX.note.innerHTML = m.note.map(t => `<span>${fmt(t)}</span>`).join("");
  if (m.plan) {
    applyPlan(m.plan);
    $(".runs").innerHTML = "Example plan of this scene, from layer 0 up: " + m.legend.map(([c, t]) =>
      `<span><b class="${c}"></b>${esc(t)}</span>`).join(" ");
  }
  ROOT.dataset.state = "rest";
  syncCtl();
  layout();
}

// The end of a scene: the steps let go, the closing note shows.
function ended() {
  anim = null;
  if (T.seeking) { T.seeking = false; settle(); }
  stepItems().forEach(li => { li.classList.remove("now"); li.removeAttribute("aria-current"); });
  ROOT.dataset.state = "done";
  syncCtl();
  layout();
}

function showGroup(k) {
  const g = D.groups[k];
  light([k], k, null);
  g.reads.forEach(lv => nerve(g.origin, D.levels[lv].at));
  const reads = g.reads.map(lv => D.levels[lv].name).join(", ");
  BOX.group.style.setProperty("--c", color(k));
  BOX.group.innerHTML = `<span class="e">Group</span><span class="t">${esc(g.name)}</span><span>${fmt(g.what)}</span>` +
    (reads ? `<span class="r">Reads the placement plan at: ${esc(reads)}.</span>` : "");
  $("#sr").textContent = `${g.name}. ${g.what}${reads ? ` Reads the placement plan at: ${reads}.` : ""}`;
  ROOT.dataset.state = "group";
  layout();
}

function showLevel(lv) {
  const l = D.levels[lv];
  const readers = Object.values(D.groups).filter(g => g.reads.includes(lv));
  light(readers.map(g => g.key), null, lv);
  readers.forEach(g => nerve(g.origin, l.at));
  const names = readers.map(g => g.name).join(", ");
  BOX.group.style.setProperty("--c", color("stack"));
  BOX.group.innerHTML = `<span class="e">Level of the placement plan</span><span class="t">${esc(l.name)}</span>` +
    `<span>${fmt(l.what)}</span><span class="r">Read by: ${esc(names)}.</span>`;
  $("#sr").textContent = `${l.name}. ${l.what} Read by: ${names}.`;
  ROOT.dataset.state = "group";
  layout();
}

// One request played on the overview: the prompt goes in, is read through the stack once, then
// each token climbs the stack, reads the KV cache, is sampled and streams out.
// A question any model answers, which says nothing about the daemon.
const DEMO = {prompt: "What is the capital of France?", tokens: ["What", " is", " the", " capital", " of", " France", "?"],
              answer: ["The", " capital", " is", " Paris", "."]};
const ROUTES = {
  kv: "M500,172 L500,76",
  sample: "M494,74 C440,120 430,220 448,292",
  out: "M456,306 C476,240 484,190 494,146",
  down: "M494,146 C470,220 476,330 500,374",
  up: "M500,374 L500,172",
};
let anim = null;
const RUN_NAMES = {dr: "the ROCm card", dc: "the CUDA card", dh: "the host",
                   pa: "card A", pb: "card B", ph: "the host", pt: "cards A and B"};

// Lights the vertebra a pulse passes, swells the pulse while it computes, and says where.
function tracker(svg, layer, report) {
  // The screen matrix is read at each call: the plate changes size in full screen.
  const toSvg = (el, x, y) => new DOMPoint(x, y).matrixTransform(svg.getScreenCTM().inverse().multiply(el.getScreenCTM()));
  const vts = [...svg.querySelectorAll(".vt")].map(el => {
    const b = el.getBBox(), c = toSvg(el, b.x + b.width / 2, b.y + b.height / 2);
    return {el, x: c.x, y: c.y, run: el.dataset.run, t: 0};
  });
  const reach = Math.max(6, vts.length > 1 ? Math.abs(vts[1].y - vts[0].y) * 1.4 : 9);
  let busy = false;
  return (dot, x, y) => {
    const p = toSvg(layer, x, y);
    let best = null, gap = reach;
    for (const v of vts) {
      const d = Math.hypot(v.x - p.x, v.y - p.y);
      if (d < gap) { gap = d; best = v; }
    }
    if (best && !T.seeking) {
      best.el.classList.add("hot");
      const t = best.t = {}, e = CTL.epoch;
      T.wait(450).then(() => { if (best.t === t && CTL.epoch === e) best.el.classList.remove("hot"); });
    }
    dot.setAttribute("r", best ? dot.dataset.big || 8 : dot.dataset.r);
    dot.classList.toggle("busy", !!best);
    if (busy !== !!best) act(svg, "backends", "flex", busy = !!best);
    report(best ? best.el.dataset.plan || best.run : null, p.x);
  };
}

// Moves the parts `part` of the figure on `svg`, or their shapes `inner`, while their modules
// work. With `on` given the move holds until it is false, else it plays once.
const MOVES = {breathe: 900, beat: 550, speak: 400, draw: 500, glow: 700};
function act(svg, part, name, on, inner) {
  // A move that plays once is skipped while seeking: the step lands at rest.
  if (on === undefined && T.seeking) return;
  svg.querySelectorAll(`[data-part="${part}"]${inner ? " " + inner : ""}`).forEach(el => {
    if (on !== undefined) return el.classList.toggle(name, on);
    el.classList.remove(name);
    el.getBoundingClientRect();
    el.classList.add(name);
    el.moves = el.moves || {};
    clearTimeout(el.moves[name]);
    el.moves[name] = setTimeout(() => el.classList.remove(name), MOVES[name]);
  });
}

function setNow(text, busy) {
  const now = $("#now");
  if (!now) return;
  now.textContent = text;
  now.classList.toggle("busy", !!busy);
}

// A writer of 1 line of words on `layer` from `x`, `y`, at the size of the plate times `scale`.
// Each word goes after the last one, 1 space apart, measured on the text drawn.
// A token that starts with a space is 1 space after the last one, any other token touches it.
function sentence(layer, x, y, scale) {
  let at = x;
  return word => {
    if (at > x && word.startsWith(" ")) at += 7 * scale;
    const t = svgEl("text", {class: "tok", x: at, y, style: `font-size:${12 * scale}px${T.seeking ? ";animation:none" : ""}`});
    t.textContent = word.trim();
    layer.append(t);
    at += t.getComputedTextLength();
    return t;
  };
}

// `fn` after `ms` of the scene clock, unless the scene stopped meanwhile.
function later(ms, fn) {
  const e = CTL.epoch;
  T.wait(ms).then(() => { if (CTL.epoch === e) fn(); });
}

function svgEl(tag, attrs) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

function stopAnim() {
  if (anim) anim.stop = true;
  anim = null;
  $$(".anim").forEach(g => g.replaceChildren());
  $$("svg.plate").forEach(s => s.classList.remove("playing"));
  $$(".vt.hot").forEach(v => v.classList.remove("hot"));
  $$(".part").forEach(e => e.classList.remove(...Object.keys(MOVES), "held", "flex", "churn"));
  $$(".part *").forEach(e => e.classList.remove(...Object.keys(MOVES)));
  T.seeking = false;
  CTL.epoch++;
  ROOT.classList.remove("seeking");
  if (T.paused) T.pause(false);
}

// The 3 cluster modes, played on the cluster plate.
const MODES = D.modes;

async function playCluster(mode) {
  show();
  const k = kit("cluster", MODES[mode]), run = k.run;
  const R = D.cluster, svg = $("#v-cluster"), layer = $("#v-cluster .anim");
  svg.classList.add("playing");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const pause = ms => T.wait(reduce ? Math.max(ms, 500) : ms);
  const dot = (cls, at) => {
    const r = cls === "kv" ? 7 : cls === "small" ? 3.5 : 4.5;
    const d = svgEl("circle", {class: `pulse ${cls}`, r, "data-r": r, "data-big": 8, cx: at[0], cy: at[1]});
    layer.append(d);
    return d;
  };
  const M = MODES[mode];
  let base = M.now[0];
  const mid = (R.a_mouth[0] + R.b_mouth[0]) / 2;
  const track = tracker(svg, layer, (runOn, x) =>
    setNow(runOn ? `Node ${x < mid ? "A" : "B"} computing on ${RUN_NAMES[runOn]}` : base, !!runOn));
  const step = i => {
    stepItems().forEach((li, j) => li.classList.toggle("now", j === i));
    base = k.base = M.now[i];
    setNow(base, false);
    stepped();
  };
  const move = (d, route, speed) => new Promise(done => {
    const path = svgEl("path", {class: "apath", d: R[route]});
    layer.append(path);
    const len = path.getTotalLength();
    const main = !d.classList.contains("small") && !d.classList.contains("kv");
    if (route.startsWith("hands")) k.act("cluster", "glow");
    const put = k => {
      const p = path.getPointAtLength(len * k);
      d.setAttribute("cx", p.x); d.setAttribute("cy", p.y);
      if (main) track(d, p.x, p.y);
    };
    if (reduce) { put(1); path.remove(); return T.wait(500).then(done); }
    const t0 = T.now(), dur = len / speed * 1000;
    const tick = now => {
      if (run.stop) { path.remove(); return done(); }
      const k = Math.min(1, (now - t0) / dur);
      put(k);
      if (k < 1) T.raf(tick); else { path.remove(); done(); }
    };
    T.raf(tick);
  });
  // The answer is written word after word on 1 line between the heads of the 2 nodes, at the
  // height of their eyes, and stays until the next scene.
  const answerLine = sentence(layer, R.a_mouth[0] + 66 * R.s, R.a_mouth[1] - 30 * R.s, R.s);
  const emit = n => {
    answerLine(DEMO.answer[n]);
    const a = $("#answer"); if (a) a.textContent += DEMO.answer[n];
    const c = $("#count"); if (c) c.textContent = `token ${n + 1} of ${DEMO.answer.length}`;
  };
  const N = DEMO.answer.length;
  // The names of the story: the verbs of the kit, and the helpers above.
  const host = AnaHost.scenes(k, {DEMO, RUN_NAMES, applyPlan: null, setNow, later, select: $});
  Object.assign(host.names, {
    R, N, step, wait: pause, kitWait: ms => k.pause(ms), cdot: dot, ride: move, emitToken: emit, track,
    travel: (d, pts, speed, tracked) => (tracked ? k.via(d, pts, speed, track) : k.via(d, pts, speed)),
    follow: (ref, x) => ({follow: ref, x, track}),
    say: text => { base = k.base = text; setNow(text, false); },
    setBase: text => { base = text; },
    now: text => setNow(text, false),
    toggleClass: (cls, things, on) => things.forEach(v => v.el.classList.toggle(cls, on)),
    clearLayer: () => layer.replaceChildren(),
    stopPlaying: () => svg.classList.remove("playing"),
    complete: ended,
  });
  try { await AnaScript.run(ROOT.anaScenes, mode, host); } catch (e) { console.error(e); ended(); }
}

// The scenes of the plates between the overview and the cluster: a kit
// their players share, and one player for each.
const sceneOf = (p, key) => D.scenes[p].find(s => s.key === key) || D.scenes[p][0];

// What every player draws with, in the coordinates of the figure on plate `p`. Sizes are given
// as on the layer stack plate and scaled to the plate.
function kit(p, s) {
  const svg = $(`#v-${p}`), layer = $(`#v-${p} .anim`), run = anim = {stop: false};
  svg.classList.add("playing");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const z = 1.3 / layer.getCTM().a;
  const k = {
    svg, layer, run, z, base: s.now[0],
    pause: ms => T.wait(ms),
    el(tag, attrs, text) {
      const e = svgEl(tag, attrs);
      if (text != null) e.textContent = text;
      layer.append(e);
      return e;
    },
    text(x, y, t, cls = "cardlab", anchor = "start", size = 9) {
      return k.el("text", {class: cls, x, y, "text-anchor": anchor, style: `font-size:${size * z}px`}, t);
    },
    dot(x, y, r = 4.5, cls = "") {
      return k.el("circle", {class: `pulse ${cls}`, r: r * z, "data-r": r * z, "data-big": 8 * z, cx: x, cy: y});
    },
    pos: d => [+d.getAttribute("cx"), +d.getAttribute("cy")],
    act: (part, name, on, inner) => act(svg, part, name, on, inner),
    // The centre of a drawn element in the coordinates of the animation layer.
    centre(el) {
      const b = el.getBBox();
      const c = new DOMPoint(b.x + b.width / 2, b.y + b.height / 2)
        .matrixTransform(layer.getScreenCTM().inverse().multiply(el.getScreenCTM()));
      return [c.x, c.y];
    },
    // The vertebrae from layer 0 up, each with its centre and the device of its run.
    vts: () => [...svg.querySelectorAll(".vt")].map(el => {
      const [x, y] = k.centre(el);
      return {el, x, y, run: el.dataset.run};
    }).sort((a, b) => b.y - a.y).map((v, n) => ({...v, n})),
    // Every dot of `moves` from where it is to its target together, in `ms`.
    glide: (moves, ms, each) => new Promise(done => {
      const from = moves.map(([d]) => k.pos(d));
      const put = t => moves.forEach(([d, [x, y]], i) => {
        const [x0, y0] = from[i], e = t < .5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t);
        const cx = x0 + (x - x0) * e, cy = y0 + (y - y0) * e;
        d.setAttribute("cx", cx); d.setAttribute("cy", cy);
        if (each) each(d, cx, cy);
      });
      if (reduce) { put(1); return T.wait(ms).then(done); }
      const t0 = T.now();
      const tick = now => {
        if (run.stop) return done();
        const t = Math.min(1, (now - t0) / ms);
        put(t);
        if (t < 1) T.raf(tick); else done();
      };
      T.raf(tick);
    }),
    // One dot through the points `pts` in turn, at `speed` figure units a second.
    async via(d, pts, speed, each) {
      for (const pt of pts) {
        const [x, y] = k.pos(d);
        await k.glide([[d, pt]], Math.max(60, Math.hypot(pt[0] - x, pt[1] - y) / speed * 1000), each);
        if (run.stop) return;
      }
    },
    swell(ds, on) {
      ds.forEach(d => { d.setAttribute("r", on ? d.dataset.big : d.dataset.r); d.classList.toggle("busy", on); });
    },
    async compute(ds, ms, text) {
      k.swell(ds, true); setNow(text || k.base, true);
      await k.pause(ms);
      // A stopped scene leaves the Now line to the scene that replaced it.
      if (run.stop) return;
      k.swell(ds, false); setNow(k.base, false);
    },
    // Lights the vertebrae `list` one after the other, as a wave; a seek lands with none lit.
    async wave(list, each = 30) {
      if (T.seeking) return;
      for (const v of list) {
        v.el.classList.add("hot");
        const e = CTL.epoch;
        T.wait(300).then(() => { if (CTL.epoch === e) v.el.classList.remove("hot"); });
        await k.pause(each);
        if (run.stop) return;
      }
    },
    step(i) {
      stepItems().forEach((li, j) => li.classList.toggle("now", j === i));
      k.base = s.now[i];
      setNow(k.base, false);
      stepped();
    },
    row(i, t) { const e = $(`#row-${i}`); if (e) e.textContent = t; },
    emit(x, y, t) {
      const e = k.el("text", {class: "tok", x, y, style: `font-size:${12 * z}px${T.seeking ? ";animation:none" : ""}`}, t);
      setTimeout(() => e.remove(), 1700);
    },
  };
  k.track = tracker(svg, layer, r => setNow(r ? `Computing on ${RUN_NAMES[r]}` : k.base, !!r));
  return k;
}

const PLAYERS = {







};

function rest() {
  stopAnim();
  CTL.count = 0;
  light([], null, null);
  setScene(CTL.key);
}

ROOT.addEventListener("click", e => {
  if (e.target.closest(scoped("#play"))) {
    if (!anim) start(CTL.key, 0, false); else T.pause(!T.paused);
    return;
  }
  if (e.target.closest(scoped("#next"))) { next(); return; }
  if (e.target.closest(scoped("#prev"))) { prev(); return; }
  const dot = e.target.closest(".dots button");
  if (dot) { start(CTL.key, 0, true, +dot.dataset.step); return; }
  const sc = e.target.closest(".modes button");
  if (sc) { start(sc.dataset.scene, 0, false); return; }
  if (e.target.closest("svg.plate") && !e.target.closest("[data-group],[data-level]")) { rest(); return; }
  const t = e.target.closest("[data-group],[data-level]");
  if (!t) return;
  stopAnim();
  if (t.dataset.group) showGroup(t.dataset.group);
  else showLevel(t.dataset.level);
});
ROOT.addEventListener("keydown", e => {
  if (e.key === "Escape") {
    if (fsOn()) { if (ROOT.classList.contains("fs-css")) fsToggle(); return; }
    rest(); return;
  }
  const g = e.target.closest && e.target.closest("[data-group],[data-level]");
  if (g && g.tagName.toLowerCase() === "g" && (e.key === "Enter" || e.key === " ")) {
    e.preventDefault(); stopAnim();
    if (g.dataset.group) showGroup(g.dataset.group); else showLevel(g.dataset.level);
    return;
  }
  if (fsOn()) {
    const onButton = e.target.closest && e.target.closest("button,[role=button]");
    if (e.key === "ArrowRight") { e.preventDefault(); next(); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); prev(); return; }
    if (e.key === " " && !onButton) { e.preventDefault(); $("#play").click(); return; }
    if (e.key === "Tab") {
      const all = [...ROOT.querySelectorAll('button:not([disabled]):not([hidden]),[tabindex="0"]')];
      const first = all[0], last = all.at(-1);
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }
  }
});

// Full screen: the Fullscreen API on the plate, else the plate fixed over the window.
const fsBtn = $(".fs");
const fsOn = () => document.fullscreenElement === ROOT || ROOT.classList.contains("fs-css");

function fsSync() {
  const on = fsOn();
  ROOT.classList.toggle("fs-on", on);
  document.documentElement.style.overflow = ROOT.classList.contains("fs-css") ? "hidden" : "";
  if (fsBtn) {
    const label = on ? "Exit full screen" : "Full screen";
    fsBtn.setAttribute("aria-pressed", String(on));
    fsBtn.setAttribute("aria-label", label);
    fsBtn.querySelector("span").textContent = label;
  }
  layout();
  if (on) { const play = $("#play"); if (play) play.focus(); }
}

function fsToggle() {
  if (document.fullscreenElement === ROOT) { document.exitFullscreen(); return; }
  if (ROOT.classList.contains("fs-css")) { ROOT.classList.remove("fs-css"); fsSync(); return; }
  const css = () => { ROOT.classList.add("fs-css"); fsSync(); };
  if (ROOT.requestFullscreen && document.fullscreenEnabled) ROOT.requestFullscreen().catch(css);
  else css();
}

if (fsBtn) {
  fsBtn.hidden = false;
  fsBtn.addEventListener("click", fsToggle);
  document.addEventListener("fullscreenchange", fsSync);
}

// The inlays: the title of the scene, the step being played, the Now box, the closing note and
// a group, each in a free zone of the plate.
// They are hidden from screen readers: the live region announces the step and the group.
const STAGE = $(".scroll"), SVG = $(`#v-${plate}`);
const BOX = {};
for (const k of ["title", "step", "now", "note", "group"]) {
  const e = document.createElement("div");
  e.className = `inlay inlay-${k}`;
  e.setAttribute("aria-hidden", "true");
  STAGE.append(e);
  BOX[k] = e;
}
const SHOWN = {rest: ["title"], play: ["title", "step", "now"], done: ["title", "now", "note"], group: ["title", "group"]};

// The zones of the plate, from its data, else the corners of its drawing.
function zones() {
  if (D.inlay && D.inlay[plate]) return D.inlay[plate];
  const v = SVG.viewBox.baseVal, w = v.width / 3;
  return {title: [[v.x + 8, 8, v.x + w, 90]], step: [[v.x + 8, v.height * .7, v.x + w, v.height - 8]],
          now: [[v.x + v.width - w, v.height * .7, v.x + v.width - 8, v.height - 8]]};
}

// The space in pixels between a box and the edge of its zone, which touches the drawing.
const GAP = 12;

// Box `e` on the first zone that no lit part of the figure overlaps.
function fitBox(e, list, above) {
  const lit = [...SVG.querySelectorAll(".part.lit .t,.part.sel .t,.vt.hot")].map(x => x.getBoundingClientRect());
  const put = z => {
    const m = SVG.getScreenCTM(), r = STAGE.getBoundingClientRect();
    const a = new DOMPoint(z[0], z[1]).matrixTransform(m), b = new DOMPoint(z[2], z[3]).matrixTransform(m);
    const w = b.x - a.x - 2 * GAP;
    let top = a.y - r.top + GAP;
    // In the column of the box above, this box starts under it.
    if (above) {
      const u = above.getBoundingClientRect();
      if (u.left < b.x && a.x < u.right && u.bottom - r.top + GAP / 2 > top) top = u.bottom - r.top + GAP / 2;
    }
    e.style.left = `${a.x - r.left + GAP}px`;
    e.style.top = `${top}px`;
    e.style.width = `${w}px`;
    e.style.maxHeight = `${b.y - r.top - GAP - top}px`;
    e.style.fontSize = `${Math.max(12, Math.min(16, w / 16))}px`;
  };
  for (const z of list) {
    put(z);
    const b = e.getBoundingClientRect();
    if (!lit.some(l => l.left < b.right && b.left < l.right && l.top < b.bottom && b.top < l.bottom)) return;
  }
  put(list[0]);
}

// A box whose text overflows its zone drops its eyebrow line.
function tighten(e) {
  e.classList.remove("tight");
  if (e.scrollHeight > e.clientHeight + 1) e.classList.add("tight");
}

let queued = false;
function layout() {
  if (!queued) { queued = true; requestAnimationFrame(place); }
}
function place() {
  queued = false;
  const Z = zones(), shown = SHOWN[ROOT.dataset.state || "rest"] || [];
  const j = stepItems().findIndex(li => li.classList.contains("now"));
  for (const [k, e] of Object.entries(BOX)) {
    const on = shown.includes(k) && (k !== "step" || j >= 0) && (k !== "note" || !!e.textContent);
    e.classList.toggle("on", on);
    // A hidden box is placed too: it never holds space outside the figure.
    const over = {step: "title", group: "title", note: "now"}[k];
    fitBox(e, Z[k] || Z.step, over && BOX[over].classList.contains("on") ? BOX[over] : null);
    if (on) tighten(e);
  }
}
new MutationObserver(layout).observe(BOX.now, {subtree: true, childList: true, characterData: true});
// The figure changes size when the page gains a scroll bar or enters full screen.
new ResizeObserver(layout).observe(SVG);

// The address names a scene: `#key`.
// The query `?scenes=a,b` limits the scenes offered.
const only = new URLSearchParams(location.search).get("scenes");
if (only) {
  const keep = only.split(",").filter(k => D.order.includes(k));
  if (keep.length) D.order = keep;
}
ROOT.classList.add("js");
$(".bar").hidden = false;
const asked = location.hash.slice(1);
CTL.key = D.order.includes(asked) ? asked : D.order[0];
show();
rest();
addEventListener("hashchange", () => {
  const k = location.hash.slice(1);
  if (D.order.includes(k)) { CTL.key = k; rest(); ROOT.scrollIntoView(); }
});
if (asked && D.order.includes(asked)) ROOT.scrollIntoView();
})();
