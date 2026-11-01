// Starts the engine of the anatomy plate when a page carries one.
// The plate is a `.ana-plate` element. Its drawing is a separate SVG file named by `data-plate-src`
// on the element that holds it. Its data is JSON next to it. Its scenes are in a text file named by
// `data-scenes`, in the language of anatomy-script.js. The attribute `data-engine` names the engine
// file: scenes, overview or cluster.
(() => {
  const root = document.querySelector(".ana-plate[data-engine]");
  if (!root) return;
  const base = document.currentScript.src.replace(/[^/]*$/, "");
  const holder = root.querySelector("[data-plate-src]");
  const load = src => new Promise((done, fail) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = done;
    script.onerror = fail;
    document.body.append(script);
  });
  const text = url => fetch(new URL(url, location.href)).then(r => {
    if (!r.ok) throw new Error(r.status);
    return r.text();
  });
  const drawing = holder ? text(holder.dataset.plateSrc).then(svg => { holder.innerHTML = svg; }) : Promise.resolve();
  const scenes = text(root.dataset.scenes);
  Promise.all([drawing, scenes])
    .then(async ([, source]) => {
      await load(base + "anatomy-script.js");
      await load(base + "anatomy-host.js");
      try { root.anaScenes = AnaScript.parse(source); } catch (e) { console.error(e); return; }
      await load(base + "anatomy-" + root.dataset.engine + ".js");
    })
    .catch(() => { if (holder && !holder.firstChild) holder.textContent = "This plate could not be loaded."; });
})();
