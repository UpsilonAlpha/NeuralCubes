// Neural CA stepping on the GPU with WebGPU compute shaders. The CPU universe (Rust/wasm)
// stays the source of truth for drawing, raycasting and editing: after GPU steps the state
// is read back into it, and after an edit it is uploaded again.
//
// The shader mirrors `NeuralCa::step` in crates/nc-core/src/rules/neural.rs, including the
// firing hash, so both paths make the same decisions.

const WG = 64;

function shader({ channels: C, hidden: H, kernels, channel_major, wrap }) {
  const K = kernels.length;
  const FEAT = K * C;
  const KER = 0, W1 = K * 27, B1 = W1 + H * FEAT, W2 = B1 + H, B2 = W2 + C * H;
  return /* wgsl */ `
struct Params {
  n: u32, generation: u32, fire_rate: f32, threshold: f32,
  rotate: u32, g0: u32, g1: u32, g2: u32,
  r0: vec4f, r1: vec4f, r2: vec4f, signs: vec4f,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<storage, read_write> pre: array<u32>;
@group(0) @binding(5) var<storage, read_write> post: array<u32>;

const C = ${C}u;
const H = ${H}u;
const K = ${K}u;
const FEAT = ${FEAT}u;

fn fi(k: u32, c: u32) -> u32 { return ${channel_major ? "c * K + k" : "k * C + c"}; }

// Index of the neighbor at offset (dx, dy, dz), or -1 beyond a zero boundary.
fn neighbor(x: i32, y: i32, z: i32, n: i32) -> i32 {
  ${wrap
    ? "let a = (x + n) % n; let b = (y + n) % n; let c = (z + n) % n; return (c * n + b) * n + a;"
    : "if (x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n) { return -1; } return (z * n + y) * n + x;"}
}

fn hash32(v: u32) -> u32 {
  var x = v;
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return x;
}

fn coords(i: u32) -> vec3i {
  let n = p.n;
  return vec3i(i32(i % n), i32((i / n) % n), i32(i / (n * n)));
}

fn any_alive_src(i: u32) -> u32 {
  let q = coords(i); let n = i32(p.n);
  for (var j = 0; j < 27; j++) {
    let nb = neighbor(q.x + j % 3 - 1, q.y + (j / 3) % 3 - 1, q.z + j / 9 - 1, n);
    if (nb >= 0 && src[u32(nb) * C + 3u] > p.threshold) { return 1u; }
  }
  return 0u;
}

fn any_alive_dst(i: u32) -> u32 {
  let q = coords(i); let n = i32(p.n);
  for (var j = 0; j < 27; j++) {
    let nb = neighbor(q.x + j % 3 - 1, q.y + (j / 3) % 3 - 1, q.z + j / 9 - 1, n);
    if (nb >= 0 && dst[u32(nb) * C + 3u] > p.threshold) { return 1u; }
  }
  return 0u;
}

fn cells() -> u32 { return p.n * p.n * p.n; }

@compute @workgroup_size(${WG})
fn alive_pre(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cells()) { return; }
  pre[i] = any_alive_src(i);
}

@compute @workgroup_size(${WG})
fn update(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cells()) { return; }
  if (pre[i] == 0u) {
    for (var c = 0u; c < C; c++) { dst[i * C + c] = 0.0; }
    return;
  }
  let h = hash32(i ^ hash32(p.generation ^ 0x9e3779b9u));
  if (f32(h >> 8u) / 16777216.0 >= p.fire_rate) {
    for (var c = 0u; c < C; c++) { dst[i * C + c] = src[i * C + c]; }
    return;
  }

  var feats: array<f32, FEAT>;
  let q = coords(i); let n = i32(p.n);
  for (var j = 0u; j < 27u; j++) {
    let ji = i32(j);
    let nb = neighbor(q.x + ji % 3 - 1, q.y + (ji / 3) % 3 - 1, q.z + ji / 9 - 1, n);
    if (nb < 0) { continue; }
    let base = u32(nb) * C;
    for (var k = 0u; k < K; k++) {
      let kw = w[${KER}u + k * 27u + j];
      if (kw == 0.0) { continue; }
      for (var c = 0u; c < C; c++) { feats[fi(k, c)] += kw * src[base + c]; }
    }
  }
  if (p.rotate != 0u) {
    // Express each channel's gradient in the shape's own frame: R^T g.
    let s = p.signs.xyz;
    for (var c = 0u; c < C; c++) {
      let ia = fi(p.g0, c); let ib = fi(p.g1, c); let ic = fi(p.g2, c);
      let v = vec3f(feats[ia], feats[ib], feats[ic]) * s;
      let r = v.x * p.r0.xyz + v.y * p.r1.xyz + v.z * p.r2.xyz;
      feats[ia] = r.x * s.x; feats[ib] = r.y * s.y; feats[ic] = r.z * s.z;
    }
  }
  var out: array<f32, C>;
  for (var c = 0u; c < C; c++) { out[c] = src[i * C + c] + w[${B2}u + c]; }
  for (var hh = 0u; hh < H; hh++) {
    var acc = w[${B1}u + hh];
    let row = ${W1}u + hh * FEAT;
    for (var f = 0u; f < FEAT; f++) { acc += w[row + f] * feats[f]; }
    if (acc > 0.0) {
      for (var c = 0u; c < C; c++) { out[c] += w[${W2}u + c * H + hh] * acc; }
    }
  }
  for (var c = 0u; c < C; c++) { dst[i * C + c] = out[c]; }
}

@compute @workgroup_size(${WG})
fn alive_post(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cells()) { return; }
  post[i] = any_alive_dst(i);
}

@compute @workgroup_size(${WG})
fn apply_mask(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= cells()) { return; }
  if (pre[i] == 0u || post[i] == 0u) {
    for (var c = 0u; c < C; c++) { dst[i * C + c] = 0.0; }
  }
}
`;
}

export class GpuNca {
  /** Resolves to a GpuNca, or null when WebGPU isn't available. */
  static async create(params, weightsJson, size) {
    if (!navigator.gpu) return null;
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    const g = new GpuNca(device, params, weightsJson, size);
    const err = await g.compiled;
    if (err) { device.destroy(); throw new Error(err); }
    return g;
  }

  constructor(device, params, weightsJson, size) {
    this.device = device;
    this.params = params;
    this.size = size;
    this.cells = size ** 3;
    const C = params.channels;
    const m = JSON.parse(weightsJson);
    const weights = new Float32Array([...params.kernels.flat(), ...m.w1, ...m.b1, ...m.w2, ...m.b2]);

    const S = GPUBufferUsage;
    const buf = (bytes, usage) => device.createBuffer({ size: Math.max(16, bytes), usage });
    this.uniform = buf(96, S.UNIFORM | S.COPY_DST);
    this.weights = buf(weights.byteLength, S.STORAGE | S.COPY_DST);
    device.queue.writeBuffer(this.weights, 0, weights);
    const stateBytes = this.cells * C * 4;
    this.state = [0, 1].map(() => buf(stateBytes, S.STORAGE | S.COPY_SRC | S.COPY_DST));
    this.pre = buf(this.cells * 4, S.STORAGE);
    this.post = buf(this.cells * 4, S.STORAGE);
    this.readback = buf(stateBytes, S.MAP_READ | S.COPY_DST);
    this.front = 0;

    const module = device.createShaderModule({ code: shader(params) });
    this.compiled = module.getCompilationInfo().then((info) => {
      const errs = info.messages.filter((x) => x.type === "error");
      return errs.length ? errs.map((x) => `${x.lineNum}: ${x.message}`).join("\n") : null;
    });
    const entry = (type) => ({ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layout = device.createBindGroupLayout({
      entries: ["uniform", "read-only-storage", "read-only-storage", "storage", "storage", "storage"]
        .map((t, i) => ({ ...entry(t), binding: i })),
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.pipelines = ["alive_pre", "update", "alive_post", "apply_mask"].map((entryPoint) =>
      device.createComputePipeline({ layout: pl, compute: { module, entryPoint } }));
    // groups[k] reads state[k] and writes state[1 - k].
    this.groups = [0, 1].map((k) => device.createBindGroup({
      layout,
      entries: [this.uniform, this.weights, this.state[k], this.state[1 - k], this.pre, this.post]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    this.setRotation(params.rotation);
  }

  /** `r` is a row-major 3x3 matrix, or null for the trained orientation. */
  setRotation(r) {
    this.rotation = r;
  }

  upload(state) {
    this.device.queue.writeBuffer(this.state[this.front], 0, state);
  }

  /** Queues `steps` steps starting at `generation`. */
  step(steps, generation) {
    const { device, params } = this;
    const groups = Math.ceil(this.cells / WG);
    const [g0, g1, g2] = params.gradient_kernels ?? [0, 0, 0];
    const r = this.rotation;
    for (let s = 0; s < steps; s++) {
      const u = new ArrayBuffer(96);
      const ui = new Uint32Array(u), uf = new Float32Array(u);
      ui[0] = this.size; ui[1] = (generation + s) >>> 0; uf[2] = params.fire_rate; uf[3] = params.alive_threshold;
      ui[4] = r && params.gradient_kernels ? 1 : 0; ui[5] = g0; ui[6] = g1; ui[7] = g2;
      if (r) for (let i = 0; i < 3; i++) uf.set(r[i], 8 + 4 * i);
      uf.set(params.gradient_signs, 20);
      device.queue.writeBuffer(this.uniform, 0, u);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setBindGroup(0, this.groups[this.front]);
      for (const pipe of this.pipelines) {
        pass.setPipeline(pipe);
        pass.dispatchWorkgroups(groups);
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      this.front = 1 - this.front;
    }
  }

  /** Reads the current state back into a Float32Array. */
  async read() {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.state[this.front], 0, this.readback, 0, this.readback.size);
    this.device.queue.submit([enc.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    return out;
  }

  destroy() {
    this.device.destroy();
  }
}
