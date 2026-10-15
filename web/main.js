import init, { Universe } from "./pkg/nc_web.js";
import { Renderer } from "./renderer.js";
import { GpuNca } from "./gpu.js";

// Well-known 3D rules, with seeding that suits each. Trained neural CA models are listed
// in models/index.json and go in front of these.
const PRESETS = [
  { name: "445", rule: "4/4/5/M", extent: 0.3, density: 0.3 },
  { name: "Amoeba", rule: "9-26/5-7,12-13,15/5/M", extent: 0.5, density: 0.5 },
  { name: "Architecture", rule: "4-6/3/2/M", extent: 0.2, density: 0.3 },
  { name: "Builder 1", rule: "2,6,9/4,6,8-9/10/M", extent: 0.15, density: 0.4 },
  { name: "Clouds 1", rule: "13-26/13-14,17-19/2/M", extent: 1, density: 0.5 },
  { name: "Crystal growth", rule: "0-6/1,3/2/N", extent: 0.04, density: 1 },
  { name: "Pyroclastic", rule: "4-7/6-8/10/M", extent: 0.2, density: 0.4 },
  { name: "Slow decay", rule: "1,4,8,11,13-26/13-26/5/M", extent: 0.6, density: 0.5 },
  { name: "Spiky growth", rule: "0-3,7-9,11-13,18,21-22,24,26/13,17,20-26/4/M", extent: 0.3, density: 0.5 },
];

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(
  ["canvas", "stats", "preset", "rule", "apply", "error", "play", "step", "reseed", "speed", "size", "extent",
    "density", "color", "brush", "modelFile", "seedControls", "hint", "viewMode", "yaw", "pitch", "roll",
    "orientControls", "tiltControls", "orientHint", "gpuControls", "useGpu", "gpuNote"].map((id) => [id, $(id)]),
);

async function modelPresets() {
  try {
    const res = await fetch("models/index.json");
    if (res.ok) return (await res.json()).map((m) => ({ name: m.label ?? m.model, model: m.model }));
  } catch { /* fall through */ }
  return [{ name: "Bunny (neural CA, regenerates)", model: "bunny" }];
}
PRESETS.unshift(...(await modelPresets()));

await init();
const renderer = new Renderer(ui.canvas);
let universe = null;
let model = null; // { name, json } when running a neural CA
let running = true;
let dirty = true;
let tool = "orbit";
let gpu = null; // GpuNca stepping the neural CA, when WebGPU is available and enabled
let gpuEpoch = 0; // bumped whenever the CPU state changes, so stale GPU readbacks are dropped
let gpuBusy = false;

// ---- URL hash: #rule=4/4/5/M&size=48 or #model=bunny&size=24 ----

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { rule: p.get("rule"), model: p.get("model"), size: Number(p.get("size")) || null };
}

function writeHash() {
  const p = model && !model.local
    ? new URLSearchParams({ model: model.name, size: universe.size() })
    : new URLSearchParams({ rule: universe.notation(), size: universe.size() });
  try { history.replaceState(null, "", "#" + p); } catch { /* sandboxed viewers may refuse */ }
}

// ---- building universes ----

function bindOutput(input, out, fmt = (v) => v) {
  const update = () => ($(out).textContent = fmt(input.value));
  input.addEventListener("input", update);
  update();
}

function setSize(n) {
  ui.size.value = n;
  ui.size.dispatchEvent(new Event("input"));
}

function reseed() {
  universe.seed(Number(ui.extent.value), Number(ui.density.value), (Math.random() * 2 ** 32) >>> 0);
  changed();
}

/** Call after any CPU-side change to the state, so the GPU copy follows it. */
function changed() {
  dirty = true;
  gpuEpoch++;
  if (gpu) {
    gpu.setRotation(JSON.parse(universe.gpu_params()).rotation);
    gpu.upload(universe.state());
  }
}

async function startGpu() {
  gpu?.destroy();
  gpu = null;
  const params = universe.family() === "neural" && universe.gpu_params();
  ui.gpuControls.hidden = !params || !navigator.gpu;
  if (!params || !navigator.gpu || !ui.useGpu.checked) return;
  const mine = universe;
  try {
    const g = await GpuNca.create(JSON.parse(params), model.json, universe.size());
    if (universe !== mine || !ui.useGpu.checked) return g?.destroy();
    gpu = g;
    ui.gpuNote.textContent = g ? "" : "No GPU adapter, so the CPU is running the model.";
    changed();
  } catch (e) {
    ui.gpuNote.textContent = `WebGPU failed, so the CPU is running the model: ${e.message ?? e}`;
  }
}

function install(next) {
  universe?.free();
  universe = next;
  ui.rule.value = universe.notation();
  ui.error.textContent = "";
  ui.seedControls.hidden = universe.family() === "neural";
  showOrientation();
  reseed();
  writeHash();
  startGpu();
}

const ORIENT_HINTS = {
  none: "This model was trained upright only, so other orientations may not grow cleanly.",
  yaw: "Trained at every turn about the vertical axis. Reseed to grow at the new angle.",
  full: "Trained at every orientation. Reseed to grow at the new angle.",
};

function showOrientation() {
  const mode = universe.orientation_mode();
  ui.orientControls.hidden = !mode;
  if (!mode) return;
  ui.tiltControls.hidden = mode !== "full";
  ui.orientHint.textContent = ORIENT_HINTS[mode] ?? "";
  applyOrientation();
}

function applyOrientation() {
  const rad = (el) => (Number(el.value) * Math.PI) / 180;
  universe.set_orientation(rad(ui.yaw), rad(ui.pitch), rad(ui.roll));
  if (gpu) gpu.setRotation(JSON.parse(universe.gpu_params()).rotation);
}

function showError(e) {
  ui.error.textContent = e?.message ?? String(e);
}

function buildRule(rule) {
  if (rule.trim().startsWith("nca:")) return loadModel(rule.trim().slice(4));
  try {
    model = null;
    install(new Universe(rule, Number(ui.size.value)));
  } catch (e) {
    showError(e);
  }
}

async function loadModel(name, json = null, size = null) {
  try {
    if (!json) {
      const res = await fetch(`models/${encodeURIComponent(name)}.json`);
      if (!res.ok) throw new Error(`No model named "${name}" (models/${name}.json: ${res.status})`);
      json = await res.text();
    }
    const grid = JSON.parse(json).grid;
    if (size) setSize(size);
    else if (grid) setSize(grid);
    model = { name, json, local: !!json && name.endsWith(".json") };
    install(Universe.from_model(json, Number(ui.size.value)));
  } catch (e) {
    showError(e);
  }
}

function applyPreset(p, size = null) {
  if (p.model) return loadModel(p.model, null, size);
  ui.extent.value = p.extent;
  ui.density.value = p.density;
  ui.extent.dispatchEvent(new Event("input"));
  ui.density.dispatchEvent(new Event("input"));
  buildRule(p.rule);
}

function rebuild() {
  if (model) install(Universe.from_model(model.json, Number(ui.size.value)));
  else buildRule(universe.notation());
}

function setRunning(on) {
  running = on;
  ui.play.textContent = on ? "Pause" : "Play";
  dirty = true;
}

// ---- controls ----

for (const [i, p] of PRESETS.entries()) {
  ui.preset.add(new Option(p.model ? p.name : `${p.name}  (${p.rule})`, i));
}
ui.preset.add(new Option("Custom", "custom"));

bindOutput(ui.speed, "speedOut", (v) => `${v} steps/s`);
bindOutput(ui.size, "sizeOut", (v) => `${v}³`);
bindOutput(ui.extent, "extentOut", (v) => `${Math.round(v * 100)}%`);
bindOutput(ui.density, "densityOut", (v) => `${Math.round(v * 100)}%`);
for (const k of ["yaw", "pitch", "roll"]) {
  bindOutput(ui[k], `${k}Out`, (v) => `${v}°`);
  ui[k].addEventListener("input", () => { if (universe) applyOrientation(); });
}
bindOutput(ui.brush, "brushOut", (v) => (Number(v) <= 0.5 ? "1 cell" : `radius ${v}`));

ui.preset.addEventListener("change", () => {
  if (ui.preset.value !== "custom") applyPreset(PRESETS[ui.preset.value]);
});
ui.apply.addEventListener("click", () => { ui.preset.value = "custom"; buildRule(ui.rule.value); });
ui.rule.addEventListener("keydown", (e) => { if (e.key === "Enter") ui.apply.click(); });
ui.play.addEventListener("click", () => setRunning(!running));
ui.step.addEventListener("click", () => { setRunning(false); universe.step(1); changed(); });
ui.useGpu.addEventListener("change", startGpu);
ui.reseed.addEventListener("click", reseed);
ui.size.addEventListener("change", rebuild);
ui.color.addEventListener("change", () => { renderer.colorMode = Number(ui.color.value); });
ui.viewMode.addEventListener("change", () => { renderer.mode = ui.viewMode.value; dirty = true; });
ui.modelFile.addEventListener("change", async () => {
  const file = ui.modelFile.files[0];
  if (!file) return;
  ui.preset.value = "custom";
  await loadModel(file.name, await file.text());
  ui.modelFile.value = "";
});

const HINTS = {
  orbit: "Drag to orbit, scroll or pinch to zoom.",
  add: "Click or drag on the surface to add material. Right-drag orbits.",
  remove: "Click or drag to remove material. Right-drag orbits.",
};
function setTool(t) {
  tool = t;
  for (const b of document.querySelectorAll("[data-tool]")) b.setAttribute("aria-pressed", String(b.dataset.tool === t));
  ui.hint.textContent = HINTS[t];
  ui.canvas.dataset.tool = t;
  renderer.preview = [];
}
for (const b of document.querySelectorAll("[data-tool]")) b.addEventListener("click", () => setTool(b.dataset.tool));
addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === " ") { e.preventDefault(); setRunning(!running); }
  else if (e.key === "1") setTool("orbit");
  else if (e.key === "2") setTool("add");
  else if (e.key === "3") setTool("remove");
  else if (e.key === ".") ui.step.click();
});

// ---- cursor raycasting ----

/** Grid cell the current tool would act on under the cursor, or null. */
function target(clientX, clientY) {
  const { origin, dir } = renderer.rayAt(clientX, clientY);
  const hit = universe.raycast(...origin, ...dir);
  if (!hit) return null;
  const [x, y, z, nx, ny, nz] = hit;
  return tool === "add" ? [x + nx, y + ny, z + nz] : [x, y, z];
}

function brushCells([cx, cy, cz]) {
  const r = Math.max(Number(ui.brush.value), 0.5), n = universe.size(), out = [];
  const R = Math.ceil(r);
  for (let z = cz - R; z <= cz + R; z++)
    for (let y = cy - R; y <= cy + R; y++)
      for (let x = cx - R; x <= cx + R; x++)
        if (x >= 0 && y >= 0 && z >= 0 && x < n && y < n && z < n &&
            (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r * r) out.push([x, y, z]);
  return out;
}

function applyTool(cell) {
  const r = Number(ui.brush.value);
  const n = tool === "add" ? universe.fill(...cell, r) : universe.erase(...cell, r);
  if (n) changed();
}

function hover(e) {
  if (tool === "orbit") return;
  const cell = target(e.clientX, e.clientY);
  renderer.preview = cell ? brushCells(cell) : [];
  renderer.previewColor = tool === "add" ? [0.45, 0.9, 0.55, 0.35] : [1, 0.35, 0.3, 0.4];
}

const pointers = new Map();
let pinch = 0;
let painting = false;
let lastPainted = "";
const c = ui.canvas;
c.addEventListener("contextmenu", (e) => e.preventDefault());
c.addEventListener("pointerdown", (e) => {
  c.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  painting = tool !== "orbit" && e.button === 0 && pointers.size === 1;
  if (painting) {
    const cell = target(e.clientX, e.clientY);
    if (cell) { applyTool(cell); lastPainted = cell.join(); }
  }
});
const end = (e) => {
  pointers.delete(e.pointerId);
  pinch = 0;
  painting = false;
  lastPainted = "";
};
c.addEventListener("pointerup", end);
c.addEventListener("pointercancel", end);
c.addEventListener("pointerleave", () => { if (!pointers.size) renderer.preview = []; });
c.addEventListener("pointermove", (e) => {
  const prev = pointers.get(e.pointerId);
  if (!prev) return hover(e);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) renderer.zoom(pinch / d);
    pinch = d;
    renderer.orbit((e.clientX - prev[0]) / 2, (e.clientY - prev[1]) / 2);
    return;
  }
  if (painting) {
    const cell = target(e.clientX, e.clientY);
    // Erasing reveals new surface under the cursor, so only repaint when the target cell changes.
    if (cell && cell.join() !== lastPainted) { applyTool(cell); lastPainted = cell.join(); }
    hover(e);
    return;
  }
  renderer.orbit(e.clientX - prev[0], e.clientY - prev[1]);
});
c.addEventListener("wheel", (e) => { e.preventDefault(); renderer.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });

// ---- start-up ----

setTool("orbit");
const fromHash = readHash();
const hashSize = fromHash.size ? Math.min(128, Math.max(16, fromHash.size)) : null;
if (hashSize) setSize(hashSize);
const canonical = (rule) => { try { return Universe.validate(rule); } catch { return null; } };
if (fromHash.model) {
  const i = PRESETS.findIndex((p) => p.model === fromHash.model);
  ui.preset.value = i >= 0 ? i : "custom";
  await loadModel(fromHash.model, null, hashSize);
} else if (fromHash.rule) {
  const i = PRESETS.findIndex((p) => p.rule && canonical(p.rule) === canonical(fromHash.rule));
  ui.preset.value = i >= 0 ? i : "custom";
  if (i >= 0) applyPreset(PRESETS[i]);
  else buildRule(fromHash.rule);
} else {
  ui.preset.value = 0;
  await applyPreset(PRESETS[0]);
}
if (!universe) {
  const i = PRESETS.findIndex((p) => p.rule);
  ui.preset.value = i;
  applyPreset(PRESETS[i]);
}

function stepOnGpu(steps) {
  const g = gpu, epoch = gpuEpoch, gen = universe.generation();
  gpuBusy = true;
  g.step(steps, gen);
  g.read()
    .then((state) => {
      if (g === gpu && epoch === gpuEpoch) { universe.set_state(state, gen + steps); dirty = true; }
    })
    .catch(() => {})
    .finally(() => { gpuBusy = false; });
}

let last = performance.now();
let owed = 0;
function frame(now) {
  if (running) {
    owed += ((now - last) / 1000) * Number(ui.speed.value);
    const steps = Math.min(Math.floor(owed), 4);
    owed = Math.min(owed - steps, 1);
    if (steps > 0 && gpu) {
      if (!gpuBusy) stepOnGpu(steps);
    } else if (steps > 0) {
      universe.step(steps);
      dirty = true;
    }
  }
  last = now;
  if (dirty) {
    let drawn;
    if (renderer.mode === "smooth") {
      // Neural alpha is already graded; binary occupancy needs one blur pass to lose its steps.
      const mesh = universe.mesh(universe.family() === "neural" ? 0 : 1);
      renderer.setMesh(mesh, universe.size());
      drawn = `${(mesh.length / 33).toLocaleString()} triangles`;
    } else {
      const cells = universe.surface();
      renderer.setCells(cells, universe.size());
      drawn = `drawn ${(cells.length / 8).toLocaleString()}`;
    }
    ui.stats.textContent =
      `${universe.family()}  ${universe.notation()}${gpu ? "  on GPU" : ""}${running ? "" : "  (paused)"}\n` +
      `step ${universe.generation()}  ·  population ${universe.population().toLocaleString()}  ·  ${drawn}`;
    dirty = false;
  }
  renderer.draw();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
