// What a scene written in the language can see and do, over the kit of the engine.
//
// `AnaHost.scenes(k, env)` gives the names of a scene of the engine of scenes: the drawing and its
// layers, the verbs of the kit, and a few functions to read and change what is drawn. `env` holds
// what the engine owns: `DEMO`, `RUN_NAMES`, `applyPlan`, `setNow`, `later` and `select`.
//
//   values   layers   base   top   z   DEMO   RUN_NAMES
//   verbs    step(i) wait(ms) say(text, busy) label(x, y, text, class= anchor= size=) dot(x, y, r, class)
//            shape(tag, {attributes}) glideAll(moves, ms, each) travel(dot, points, speed, tracked)
//            compute(dots, ms, text) wave(layers, ms) swell(dots, on) light(part, name, on, inner)
//            emit(point, text) row(i, text) plan([[plan, count], …]) remove(things)
//   reading  attr(thing, name) pos(dot) centre(thing) select(selector) query(selector) textWidth(text)
//   changing addClass(class, things) dropClass(class, things) retext(thing, text) setAttr(thing, name, v)
//            style(thing, name, value)
// A layer stands for its drawn element wherever a class, an attribute or a text is changed.
window.AnaHost = {
  scenes(k, env) {
    const layers = k.vts();
    const flat = x => (Array.isArray(x) ? x.flatMap(flat) : [x]);
    const node = x => (x && x.el && x.el.classList ? x.el : x);
    const orNone = x => (x === null ? undefined : x);
    return {
      stopped: () => k.run.stop,
      later: env.later,
      names: {
        layers, base: layers.length ? layers[0].y + 14 : null, top: layers.at(-1) || null, z: k.z, DEMO: env.DEMO, RUN_NAMES: env.RUN_NAMES,
        step: i => k.step(i),
        wait: ms => k.pause(ms),
        say: (text, busy) => env.setNow(text, !!busy),
        label: (x, y, text, o = {}) => k.text(x, y, text, o.class ?? "cardlab", o.anchor ?? "start", o.size ?? 9),
        labelAt: (pt, text, o = {}) => k.text(pt[0], pt[1], text, o.class ?? "cardlab", o.anchor ?? "start", o.size ?? 9),
        dot: (x, y, r, cls) => k.dot(x, y, orNone(r), orNone(cls)),
        dotAt: (pt, r, cls) => k.dot(pt[0], pt[1], orNone(r), orNone(cls)),
        shape: (tag, attrs, text) => k.el(tag, attrs, orNone(text)),
        glide: (moves, ms, each) => k.glide(moves, ms, each && each.follow
          ? (d, x, y) => d === moves[0][0] && (each.track || k.track)(each.follow, each.x, y)
          : orNone(each)),
        glideAll: (moves, ms, each) => k.glide(moves, ms, each && each.follow
          ? (d, x, y) => d === moves[0][0] && (each.track || k.track)(each.follow, each.x, y)
          : orNone(each)),
        follow: (ref, x) => ({follow: ref, x}),
        travel: (d, pts, speed, tracked) => (tracked ? k.via(d, pts, speed, k.track) : k.via(d, pts, speed)),
        compute: (ds, ms, text) => k.compute(flat(ds), ms, orNone(text)),
        wave: (list, ms) => k.wave(list, ms),
        swell: (ds, on) => k.swell(flat(ds), on),
        light: (part, name, on, inner) => k.act(part, name, orNone(on), orNone(inner)),
        emit: (pt, text) => k.emit(pt[0], pt[1], text),
        row: (i, text) => k.row(i, text),
        plan: pairs => env.applyPlan(pairs),
        remove: things => flat(things).forEach(t => t.remove()),
        addClass: (cls, things) => flat(things).forEach(t => node(t).classList.add(cls)),
        dropClass: (cls, things) => flat(things).forEach(t => node(t).classList.remove(cls)),
        contains: (t, cls) => node(t).classList.contains(cls),
        attr: (t, name) => +node(t).getAttribute(name),
        setAttr: (t, name, v) => node(t).setAttribute(name, v),
        data: (t, name) => node(t).dataset[name],
        retext: (t, text) => { node(t).textContent = text; },
        style: (t, name, v) => (v === undefined ? node(t).style[name] : (node(t).style[name] = v)),
        pos: d => k.pos(d),
        centre: t => k.centre(node(t)),
        select: sel => env.select(sel),
        query: sel => [...k.svg.querySelectorAll(sel)],
        textWidth: t => t.getComputedTextLength(),
        box: t => t.getBBox(),
      },
    };
  },
};
