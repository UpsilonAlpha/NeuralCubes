import init, { Universe } from "./pkg/nc_web.js";
import { Renderer } from "./renderer.js";

// Well-known 3D rules, with seeding that suits each.
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
  ["canvas", "stats", "preset", "rule", "apply", "error", "play", "step", "reseed", "speed", "size", "extent", "density", "color"]
    .map((id) => [id, $(id)]),
);

await init();
const renderer = new Renderer(ui.canvas);
let universe = null;
let running = true;
let dirty = true;
let visible = 0;

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { rule: p.get("rule"), size: Number(p.get("size")) || null };
}

function writeHash() {
  const p = new URLSearchParams({ rule: universe.notation(), size: universe.size() });
  history.replaceState(null, "", "#" + p);
}

function bindOutput(input, out, fmt = (v) => v) {
  const update = () => ($(out).textContent = fmt(input.value));
  input.addEventListener("input", update);
  update();
}

function reseed() {
  universe.seed(Number(ui.extent.value), Number(ui.density.value), (Math.random() * 2 ** 32) >>> 0);
  dirty = true;
}

function build(rule) {
  try {
    const next = new Universe(rule, Number(ui.size.value));
    universe?.free();
    universe = next;
    ui.rule.value = universe.notation();
    ui.error.textContent = "";
    reseed();
    writeHash();
  } catch (e) {
    ui.error.textContent = e.message ?? String(e);
  }
}

function applyPreset(p) {
  ui.extent.value = p.extent;
  ui.density.value = p.density;
  ui.extent.dispatchEvent(new Event("input"));
  ui.density.dispatchEvent(new Event("input"));
  build(p.rule);
}

function setRunning(on) {
  running = on;
  ui.play.textContent = on ? "Pause" : "Play";
}

for (const [i, p] of PRESETS.entries()) ui.preset.add(new Option(`${p.name}  (${p.rule})`, i));
ui.preset.add(new Option("Custom", "custom"));

bindOutput(ui.speed, "speedOut", (v) => `${v} gen/s`);
bindOutput(ui.size, "sizeOut", (v) => `${v}³`);
bindOutput(ui.extent, "extentOut", (v) => `${Math.round(v * 100)}%`);
bindOutput(ui.density, "densityOut", (v) => `${Math.round(v * 100)}%`);

ui.preset.addEventListener("change", () => {
  if (ui.preset.value !== "custom") applyPreset(PRESETS[ui.preset.value]);
});
ui.apply.addEventListener("click", () => { ui.preset.value = "custom"; build(ui.rule.value); });
ui.rule.addEventListener("keydown", (e) => { if (e.key === "Enter") ui.apply.click(); });
ui.play.addEventListener("click", () => setRunning(!running));
ui.step.addEventListener("click", () => { setRunning(false); universe.step(1); dirty = true; });
ui.reseed.addEventListener("click", reseed);
ui.size.addEventListener("change", () => build(universe.notation()));
ui.color.addEventListener("change", () => { renderer.colorMode = Number(ui.color.value); });

const fromHash = readHash();
if (fromHash.size) ui.size.value = Math.min(128, Math.max(16, fromHash.size));
ui.size.dispatchEvent(new Event("input"));
const canonical = (rule) => { try { return Universe.validate(rule); } catch { return null; } };
const match = fromHash.rule ? PRESETS.findIndex((p) => canonical(p.rule) === canonical(fromHash.rule)) : -1;
if (fromHash.rule && match < 0) {
  ui.preset.value = "custom";
  build(fromHash.rule);
} else {
  ui.preset.value = Math.max(match, 0);
  applyPreset(PRESETS[ui.preset.value]);
}
if (!universe) applyPreset(PRESETS[0]);

let last = performance.now();
let owed = 0;
function frame(now) {
  if (running) {
    owed += ((now - last) / 1000) * Number(ui.speed.value);
    const steps = Math.min(Math.floor(owed), 4);
    owed = Math.min(owed - steps, 1);
    if (steps > 0) { universe.step(steps); dirty = true; }
  }
  last = now;
  if (dirty) {
    const cells = universe.surface();
    visible = cells.length / 4;
    renderer.setCells(cells, universe.size());
    ui.stats.textContent =
      `${universe.family()}  ${universe.notation()}\n` +
      `generation ${universe.generation()}  ·  population ${universe.population().toLocaleString()}  ·  drawn ${visible.toLocaleString()}`;
    dirty = false;
  }
  renderer.draw();
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
