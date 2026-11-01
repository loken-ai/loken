// The language of the scenes of a plate.
//
// A file holds scenes. A scene is a story told in lines, one statement on each. `#` starts a comment.
//
//   to mid(b)                          a procedure; it is called as a statement or inside an expression
//     return (attr(b, "x") + 5, attr(b, "y") + 5)
//   end
//
//   scene one                          the scene of that key; the procedures above it are shared
//     step 0
//     let tag = label(528, base + 30, "card A", anchor="end")
//     wait 1.3s
//     for v in layers                  repeats for every item of a list
//       if v.n < 2 then ... end
//     end
//   end
//
// Values are numbers, texts "like this", true, false, none, lists [1, 2], points (x, y), which are
// lists of 2, and objects {class: "box", x: 4}. A duration is `1.3s` or `400ms` and is a number of
// milliseconds. Expressions use `+ - * / %`, `== != < <= > >=`, `and or not`, `a if c else b`,
// `x.name`, `x[i]` (a negative index counts from the end), `x[a:b]`, calls `f(a, b, name=value)` and
// lists made by a rule: `[v for v in layers if v.run == "dc"]`.
//
// Statements:
//   let x = e            declares x                 x = e, x += e      changes it
//   x.name = e           x[i] = e                   changes a part of a thing
//   if c ... elif c ... else ... end                for x in e ... end        while c ... end
//   for [a, b] in e ...  takes a list of pairs apart
//   to f(a, b) ... end   defines a procedure        return e
//   together ... also ... end            runs the branches at the same time and waits for all
//   together for x in e ... end          runs every turn of the loop at the same time
//   start name ... end   runs a block in the background        finish name      waits for it
//   later 400ms ... end  runs a block after that delay
//   f(a, b)              calls a procedure or a verb of the engine
//
// Commands for the common verbs, which read like the story. Their words cannot name a procedure.
//   step 2                                     moves the text to its step 2
//   wait 1s                                    waits
//   say "text"  /  say "text" busy             writes the Now line
//   glide p to (x, y), q to (x, y) over 500    moves dots together           (… each f)
//   move p via (x, y), (x, y) speed 300        moves one dot through points  (… tracked)
//                                              travel(p, points, speed, tracked) does the same on a list
//   compute p, q for 700ms say "text"          makes dots compute
//   wave list 30ms                             lights the layers of a list one after the other
//   swell p, q on                              makes dots big (or `off`)
//   light "part" "name" [on | off] [in ".t"]   starts or stops an animation of the drawing
//   add "hot" to v     drop "hot" from v       a class on a thing or on a list of things
//   remove p, q                                removes things or lists of things
//   emit at (x, y) "text"                      writes a token that fades
//   row 0 "text"                               writes the text of a row of the Now box
//
// A scene stops at once when the person leaves it: no statement runs after that.
window.AnaScript = (() => {
  const KEYWORDS = new Set(["let", "if", "then", "elif", "else", "end", "for", "in", "while", "to", "return",
    "together", "also", "start", "finish", "later", "scene", "and", "or", "not", "true", "false", "none",
    "over", "via", "speed", "tracked", "each", "say", "busy", "on", "off", "at", "from"]);

  class ScriptError extends Error {}
  const fail = (line, message) => { throw new ScriptError(`line ${line}: ${message}`); };
  // Thrown when the person has left the scene; it is not an error.
  class Stop {}

  // ---- Words ------------------------------------------------------------------------------------
  function tokenize(src) {
    const out = [];
    let i = 0, line = 1, depth = 0;
    const push = (t, v) => out.push({t, v, line});
    while (i < src.length) {
      const c = src[i];
      if (c === "\n") { if (!depth) push("nl"); line++; i++; continue; }
      if (c === " " || c === "\t" || c === "\r") { i++; continue; }
      if (c === "#") { while (i < src.length && src[i] !== "\n") i++; continue; }
      if (c === '"') {
        let j = i + 1, s = "";
        while (j < src.length && src[j] !== '"') {
          if (src[j] === "\n") fail(line, "a text is not closed");
          if (src[j] === "\\") { j++; s += src[j] === "n" ? "\n" : src[j]; } else s += src[j];
          j++;
        }
        if (src[j] !== '"') fail(line, "a text is not closed");
        push("str", s); i = j + 1; continue;
      }
      const prev = out.at(-1);
      const afterValue = prev && (prev.t === "num" || prev.t === "id" || prev.t === "str" || (prev.t === "op" && (prev.v === ")" || prev.v === "]")));
      const m = /^(\d+(?:\.\d+)?|\.\d+)(ms|s)?/.exec(src.slice(i));
      if (m && (/\d/.test(c) || !afterValue)) {
        push("num", m[2] === "s" ? +m[1] * 1000 : +m[1]); i += m[0].length; continue;
      }
      const w = /^[A-Za-z_][A-Za-z_0-9]*/.exec(src.slice(i));
      if (w) { push("id", w[0]); i += w[0].length; continue; }
      const o = /^(==|!=|<=|>=|\+=|-=|[-+*\/%()\[\]{},.:=<>])/.exec(src.slice(i));
      if (!o) fail(line, `unexpected character ${c}`);
      if ("([{".includes(o[0])) depth++;
      if (")]}".includes(o[0])) depth--;
      push("op", o[0]); i += o[0].length;
    }
    push("nl"); push("eof");
    return out;
  }

  // ---- Grammar ----------------------------------------------------------------------------------
  function parse(src) {
    const toks = tokenize(src);
    let p = 0;
    const peek = () => toks[p], next = () => toks[p++];
    const isOp = v => peek().t === "op" && peek().v === v;
    const isId = v => peek().t === "id" && peek().v === v;
    const eat = v => { if (isOp(v)) { p++; return true; } return false; };
    const eatId = v => { if (isId(v)) { p++; return true; } return false; };
    const need = v => { if (!eat(v)) fail(peek().line, `expected ${v}`); };
    const needId = v => { if (!eatId(v)) fail(peek().line, `expected ${v}`); };
    const skipNl = () => { while (peek().t === "nl") p++; };
    const ident = () => { const t = next(); if (t.t !== "id" || KEYWORDS.has(t.v)) fail(t.line, `expected a name, found ${t.v ?? t.t}`); return t.v; };
    const names = () => {
      if (!eat("[")) return [ident()];
      const n = [ident()];
      while (eat(",")) n.push(ident());
      need("]");
      return n;
    };
    const lit = v => ({k: "lit", v});

    // Expressions, from the loosest to the tightest.
    const expr = () => {
      const a = orExpr();
      if (isId("if")) {
        const line = next().line;
        const c = orExpr();
        if (!eatId("else")) fail(line, "a choice needs `else`");
        return {k: "cond", c, a, b: expr(), line};
      }
      return a;
    };
    const orExpr = () => { let a = andExpr(); while (isId("or")) { const line = next().line; a = {k: "or", a, b: andExpr(), line}; } return a; };
    const andExpr = () => { let a = notExpr(); while (isId("and")) { const line = next().line; a = {k: "and", a, b: notExpr(), line}; } return a; };
    const notExpr = () => { if (isId("not")) { const line = next().line; return {k: "not", a: notExpr(), line}; } return cmpExpr(); };
    const cmpExpr = () => {
      let a = sumExpr();
      while (peek().t === "op" && ["==", "!=", "<", "<=", ">", ">="].includes(peek().v)) { const t = next(); a = {k: "bin", op: t.v, a, b: sumExpr(), line: t.line}; }
      return a;
    };
    const sumExpr = () => {
      let a = mulExpr();
      while (peek().t === "op" && (peek().v === "+" || peek().v === "-")) { const t = next(); a = {k: "bin", op: t.v, a, b: mulExpr(), line: t.line}; }
      return a;
    };
    const mulExpr = () => {
      let a = unary();
      while (peek().t === "op" && (peek().v === "*" || peek().v === "/" || peek().v === "%")) { const t = next(); a = {k: "bin", op: t.v, a, b: unary(), line: t.line}; }
      return a;
    };
    const unary = () => { if (isOp("-")) { const line = next().line; return {k: "neg", a: unary(), line}; } return postfix(); };
    const args = () => {
      const list = [], named = [];
      if (!isOp(")")) do {
        if (peek().t === "id" && toks[p + 1].t === "op" && toks[p + 1].v === "=") { const name = next().v; p++; named.push([name, expr()]); }
        else list.push(expr());
      } while (eat(","));
      need(")");
      return {list, named};
    };
    const postfix = () => {
      let a = primary();
      for (;;) {
        const line = peek().line;
        if (eat("(")) a = {k: "call", f: a, ...args(), line};
        else if (eat(".")) a = {k: "prop", a, name: next().v, line};
        else if (eat("[")) {
          let lo = null, hi = null, slice = false;
          if (!isOp(":")) lo = expr();
          if (eat(":")) { slice = true; if (!isOp("]")) hi = expr(); }
          need("]");
          a = slice ? {k: "slice", a, lo, hi, line} : {k: "index", a, i: lo, line};
        } else return a;
      }
    };
    const primary = () => {
      const t = next();
      if (t.t === "num" || t.t === "str") return lit(t.v);
      if (t.t === "id") {
        if (t.v === "true") return lit(true);
        if (t.v === "false") return lit(false);
        if (t.v === "none") return lit(null);
        if (KEYWORDS.has(t.v)) fail(t.line, `\`${t.v}\` is a word of the language`);
        return {k: "name", name: t.v, line: t.line};
      }
      if (t.t === "op" && t.v === "(") {
        const first = expr();
        if (eat(",")) { const items = [first]; do items.push(expr()); while (eat(",")); need(")"); return {k: "list", items, line: t.line}; }
        need(")"); return first;
      }
      if (t.t === "op" && t.v === "[") {
        if (eat("]")) return {k: "list", items: [], line: t.line};
        const first = expr();
        if (eatId("for")) {
          const ns = names();
          needId("in");
          const source = orExpr();
          const cond = eatId("if") ? orExpr() : null;
          need("]");
          return {k: "comp", e: first, names: ns, src: source, cond, line: t.line};
        }
        const items = [first];
        while (eat(",")) { if (isOp("]")) break; items.push(expr()); }
        need("]");
        return {k: "list", items, line: t.line};
      }
      if (t.t === "op" && t.v === "{") {
        const pairs = [];
        if (!isOp("}")) do {
          const key = next();
          need(":");
          pairs.push([key.v, expr()]);
        } while (eat(",") && !isOp("}"));
        need("}");
        return {k: "obj", pairs, line: t.line};
      }
      return fail(t.line, `unexpected ${t.v ?? t.t}`);
    };

    // Statements.
    const endStmt = () => { if (peek().t !== "nl" && peek().t !== "eof") fail(peek().line, `unexpected ${peek().v ?? peek().t}`); skipNl(); };
    const block = (...stops) => {
      const body = [];
      skipNl();
      while (!(peek().t === "id" && stops.includes(peek().v))) {
        if (peek().t === "eof") fail(peek().line, `expected ${stops.join(" or ")}`);
        body.push(statement());
        skipNl();
      }
      return body;
    };
    const list1 = () => { const items = [expr()]; while (eat(",")) items.push(expr()); return items; };
    const verb = (name, line, list) => ({k: "call", f: {k: "name", name, line}, list, named: [], line, stmt: true});
    function statement() {
      const t = peek(), line = t.line;
      if (t.t === "id") {
        switch (t.v) {
          case "let": { next(); const name = ident(); need("="); const e = expr(); endStmt(); return {k: "let", name, e, line}; }
          case "if": {
            next();
            const arms = [];
            let c = expr(); eatId("then");
            arms.push([c, block("elif", "else", "end")]);
            let other = null;
            for (;;) {
              if (eatId("elif")) { c = expr(); eatId("then"); arms.push([c, block("elif", "else", "end")]); }
              else if (eatId("else")) { other = block("end"); }
              else break;
            }
            needId("end"); endStmt();
            return {k: "if", arms, other, line};
          }
          case "for": {
            next();
            const ns = names();
            needId("in");
            const source = expr();
            const body = block("end"); needId("end"); endStmt();
            return {k: "for", names: ns, src: source, body, line};
          }
          case "while": { next(); const c = expr(); const body = block("end"); needId("end"); endStmt(); return {k: "while", c, body, line}; }
          case "to": {
            next();
            const name = ident(); need("(");
            const params = [];
            if (!isOp(")")) do params.push(ident()); while (eat(","));
            need(")");
            const body = block("end"); needId("end"); endStmt();
            return {k: "proc", name, params, body, line};
          }
          case "return": { next(); const e = (peek().t === "nl" || peek().t === "eof") ? null : expr(); endStmt(); return {k: "return", e, line}; }
          case "together": {
            next();
            if (eatId("for")) {
              const ns = names();
              needId("in");
              const source = expr();
              const body = block("end"); needId("end"); endStmt();
              return {k: "togetherfor", names: ns, src: source, body, line};
            }
            const branches = [block("also", "end")];
            while (eatId("also")) branches.push(block("also", "end"));
            needId("end"); endStmt();
            return {k: "together", branches, line};
          }
          case "start": { next(); const name = ident(); const body = block("end"); needId("end"); endStmt(); return {k: "start", name, body, line}; }
          case "finish": { next(); const name = ident(); endStmt(); return {k: "finish", name, line}; }
          case "later": { next(); const ms = expr(); const body = block("end"); needId("end"); endStmt(); return {k: "later", ms, body, line}; }
          case "step": { next(); const e = expr(); endStmt(); return verb("step", line, [e]); }
          case "wait": { next(); const e = expr(); endStmt(); return verb("wait", line, [e]); }
          case "say": { next(); const e = expr(); const busy = eatId("busy"); endStmt(); return verb("say", line, [e, lit(busy)]); }
          case "glide": {
            next();
            const moves = [];
            do { const d = expr(); needId("to"); moves.push({k: "list", items: [d, expr()], line}); } while (eat(","));
            needId("over"); const ms = expr();
            const each = eatId("each") ? expr() : lit(null);
            endStmt();
            return verb("glide", line, [{k: "list", items: moves, line}, ms, each]);
          }
          case "move": {
            next();
            const d = expr(); needId("via");
            const pts = list1();
            needId("speed"); const speed = expr(); const tracked = eatId("tracked"); endStmt();
            return verb("travel", line, [d, {k: "list", items: pts, line}, speed, lit(tracked)]);
          }
          case "compute": {
            next();
            const ds = list1(); needId("for"); const ms = expr();
            const text = eatId("say") ? expr() : lit(null);
            endStmt();
            return verb("compute", line, [{k: "list", items: ds, line}, ms, text]);
          }
          case "wave": { next(); const l = expr(); const ms = expr(); endStmt(); return verb("wave", line, [l, ms]); }
          case "swell": {
            next(); const ds = list1(); let on = true;
            if (eatId("off")) on = false; else needId("on");
            endStmt();
            return verb("swell", line, [{k: "list", items: ds, line}, lit(on)]);
          }
          case "light": {
            next(); const part = expr(), name = expr();
            let on = lit(null);
            if (eatId("off")) on = lit(false); else if (eatId("on")) on = lit(true);
            const inner = eatId("in") ? expr() : lit(null);
            endStmt();
            return verb("light", line, [part, name, on, inner]);
          }
          case "add": case "drop": {
            next(); const cls = expr(); needId(t.v === "add" ? "to" : "from");
            const things = list1(); endStmt();
            return verb(t.v === "add" ? "addClass" : "dropClass", line, [cls, {k: "list", items: things, line}]);
          }
          case "remove": { next(); const things = list1(); endStmt(); return verb("remove", line, [{k: "list", items: things, line}]); }
          case "emit": { next(); needId("at"); const at = expr(); const text = expr(); endStmt(); return verb("emit", line, [at, text]); }
          case "row": { next(); const i = expr(), text = expr(); endStmt(); return verb("row", line, [i, text]); }
          default:
        }
      }
      // An assignment, or a call used as a statement.
      const target = postfix();
      if (peek().t === "op" && (peek().v === "=" || peek().v === "+=" || peek().v === "-=")) {
        const op = next().v, e = expr(); endStmt();
        if (target.k !== "name" && target.k !== "prop" && target.k !== "index") fail(line, "this cannot be changed");
        return {k: "set", target, op, e, line};
      }
      endStmt();
      if (target.k !== "call") fail(line, "a line must do something");
      return {...target, stmt: true};
    }

    // A file: shared procedures, then scenes.
    const shared = [], scenes = {};
    skipNl();
    while (peek().t !== "eof") {
      if (eatId("scene")) {
        const key = ident();
        scenes[key] = block("end"); needId("end"); skipNl();
      } else {
        shared.push(statement()); skipNl();
      }
    }
    return {shared, scenes};
  }

  // ---- Running ----------------------------------------------------------------------------------
  class Scope {
    constructor(parent) { this.vars = new Map(); this.parent = parent; }
    find(name) { for (let s = this; s; s = s.parent) if (s.vars.has(name)) return s; return null; }
  }
  class Return { constructor(v) { this.v = v; } }

  // `+` adds numbers, joins texts and puts lists end to end.
  const add = (a, b) => (Array.isArray(a) && Array.isArray(b) ? a.concat(b) : a + b);
  const BUILTINS = {
    range: (a, b, c) => { const [lo, hi, st] = b === undefined ? [0, a, 1] : [a, b, c ?? 1]; const r = []; for (let x = lo; st > 0 ? x < hi : x > hi; x += st) r.push(x); return r; },
    len: x => x.length, floor: Math.floor, ceil: Math.ceil, round: Math.round, abs: Math.abs, min: Math.min, max: Math.max,
    enumerate: l => l.map((x, i) => [i, x]),
    num: x => +x, trim: x => x.trim(), sortBy: (l, key) => [...l].sort((a, b) => a[key] - b[key]),
    hypot: Math.hypot, random: () => Math.random(), str: x => String(x), join: (l, s = "") => l.join(s), fixed: (x, n) => x.toFixed(n),
  };

  // Runs a scene. `host.names` holds what the engine offers: values, verbs and procedures. `host.stopped`
  // tells that the person has left the scene, and `host.later(ms, f)` runs `f` after a delay.
  async function run(program, key, host) {
    const stopped = host.stopped || (() => false);
    const check = () => { if (stopped()) throw new Stop(); };
    const root = new Scope(null);
    const background = new Map();
    const lookup = (scope, name, line) => {
      const s = scope.find(name);
      if (s) return s.vars.get(name);
      if (name in host.names) return host.names[name];
      if (name in BUILTINS) return BUILTINS[name];
      return fail(line, `unknown name ${name}`);
    };
    const callable = async (f, argv, named, line) => {
      let value;
      if (f && f.proc) {
        const scope = new Scope(f.scope);
        f.proc.params.forEach((n, i) => scope.vars.set(n, i < argv.length ? argv[i] : ((named.find(([k2]) => k2 === n) || [])[1] ?? null)));
        try { await execBlock(f.proc.body, scope); value = null; } catch (e) { if (e instanceof Return) value = e.v; else throw e; }
      } else if (typeof f === "function") {
        value = await f(...argv, ...(named.length ? [Object.fromEntries(named)] : []));
      } else fail(line, "this cannot be called");
      check();
      return value;
    };
    const destructure = (scope, ns, v) => {
      if (ns.length === 1) scope.vars.set(ns[0], v);
      else ns.forEach((n, i) => scope.vars.set(n, v[i]));
    };
    async function evalExpr(e, scope) {
      switch (e.k) {
        case "lit": return e.v;
        case "name": return lookup(scope, e.name, e.line);
        case "list": { const out = []; for (const x of e.items) out.push(await evalExpr(x, scope)); return out; }
        case "obj": { const o = {}; for (const [k2, x] of e.pairs) o[k2] = await evalExpr(x, scope); return o; }
        case "neg": return -(await evalExpr(e.a, scope));
        case "not": return !(await evalExpr(e.a, scope));
        case "and": { const a = await evalExpr(e.a, scope); return a ? evalExpr(e.b, scope) : a; }
        case "or": { const a = await evalExpr(e.a, scope); return a ? a : evalExpr(e.b, scope); }
        case "cond": return (await evalExpr(e.c, scope)) ? evalExpr(e.a, scope) : evalExpr(e.b, scope);
        case "bin": {
          const a = await evalExpr(e.a, scope), b = await evalExpr(e.b, scope);
          switch (e.op) {
            case "+": return add(a, b); case "-": return a - b; case "*": return a * b; case "/": return a / b; case "%": return a % b;
            case "==": return a === b; case "!=": return a !== b; case "<": return a < b; case "<=": return a <= b; case ">": return a > b; default: return a >= b;
          }
        }
        case "prop": { const o = await evalExpr(e.a, scope); if (o == null) fail(e.line, `${e.name} of nothing`); return o[e.name]; }
        case "index": { const o = await evalExpr(e.a, scope); let i = await evalExpr(e.i, scope); if (typeof i === "number" && i < 0 && o.length) i += o.length; return o[i]; }
        case "slice": { const o = await evalExpr(e.a, scope); const lo = e.lo ? await evalExpr(e.lo, scope) : 0, hi = e.hi ? await evalExpr(e.hi, scope) : undefined; return o.slice(lo, hi); }
        case "comp": {
          const out = [];
          for (const item of await evalExpr(e.src, scope)) {
            const inner = new Scope(scope);
            destructure(inner, e.names, item);
            if (e.cond && !(await evalExpr(e.cond, inner))) continue;
            out.push(await evalExpr(e.e, inner));
          }
          return out;
        }
        case "call": {
          const f = await evalExpr(e.f, scope);
          const argv = [];
          for (const x of e.list) argv.push(await evalExpr(x, scope));
          const named = [];
          for (const [n, x] of e.named) named.push([n, await evalExpr(x, scope)]);
          return callable(f, argv, named, e.line);
        }
        default: return fail(e.line, `cannot evaluate ${e.k}`);
      }
    }
    async function execBlock(body, scope) {
      for (const s of body) {
        check();
        await exec(s, scope);
      }
    }
    async function exec(s, scope) {
      switch (s.k) {
        case "let": scope.vars.set(s.name, await evalExpr(s.e, scope)); break;
        case "set": {
          let v = await evalExpr(s.e, scope);
          if (s.op !== "=") { const cur = await evalExpr(s.target, scope); v = s.op === "+=" ? add(cur, v) : cur - v; }
          if (s.target.k === "name") {
            const owner = scope.find(s.target.name);
            if (!owner) fail(s.line, `${s.target.name} was not declared; use let`);
            owner.vars.set(s.target.name, v);
          } else if (s.target.k === "prop") (await evalExpr(s.target.a, scope))[s.target.name] = v;
          else (await evalExpr(s.target.a, scope))[await evalExpr(s.target.i, scope)] = v;
          break;
        }
        case "proc": scope.vars.set(s.name, {proc: s, scope}); break;
        case "call": await evalExpr(s, scope); break;
        case "return": throw new Return(s.e ? await evalExpr(s.e, scope) : null);
        case "if": {
          for (const [c, body] of s.arms) if (await evalExpr(c, scope)) { await execBlock(body, new Scope(scope)); return; }
          if (s.other) await execBlock(s.other, new Scope(scope));
          break;
        }
        case "while": while (await evalExpr(s.c, scope)) await execBlock(s.body, new Scope(scope)); break;
        case "for": {
          for (const item of await evalExpr(s.src, scope)) {
            const inner = new Scope(scope);
            destructure(inner, s.names, item);
            await execBlock(s.body, inner);
          }
          break;
        }
        case "together": await Promise.all(s.branches.map(b => execBlock(b, new Scope(scope)))); break;
        case "togetherfor": {
          const items = await evalExpr(s.src, scope);
          await Promise.all(items.map(item => { const inner = new Scope(scope); destructure(inner, s.names, item); return execBlock(s.body, inner); }));
          break;
        }
        case "start": background.set(s.name, execBlock(s.body, new Scope(scope)).catch(e => { if (!(e instanceof Stop)) throw e; })); break;
        case "finish": await (background.get(s.name) || fail(s.line, `nothing started as ${s.name}`)); break;
        case "later": {
          const ms = await evalExpr(s.ms, scope);
          host.later(ms, () => execBlock(s.body, new Scope(scope)).catch(e => { if (!(e instanceof Stop)) throw e; }));
          break;
        }
        default: fail(s.line, `cannot run ${s.k}`);
      }
    }
    const sceneBody = program.scenes[key];
    if (!sceneBody) fail(0, `no scene ${key}`);
    try {
      await execBlock(program.shared, root);
      await execBlock(sceneBody, new Scope(root));
    } catch (e) {
      if (e instanceof Stop) return false;
      throw e;
    }
    return true;
  }

  return {parse, run, ScriptError, Stop};
})();
