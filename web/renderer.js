// Instanced-cube WebGL2 renderer with an orbit camera. No dependencies.

const VERT = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aCell; // x, y, z, shade
uniform mat4 uViewProj;
uniform float uSize;
uniform float uGap;
out vec3 vNormal;
out vec3 vCell;
out float vShade;
void main() {
  vec3 p = aCell.xyz - vec3(uSize * 0.5 - 0.5) + aPos * (1.0 - uGap);
  gl_Position = uViewProj * vec4(p, 1.0);
  vNormal = aNormal;
  vCell = aCell.xyz / uSize;
  vShade = aCell.w;
}`;

const FRAG = `#version 300 es
precision highp float;
in vec3 vNormal;
in vec3 vCell;
in float vShade;
uniform int uColorMode; // 0 = by state, 1 = by position
out vec4 outColor;
vec3 hsv(float h, float s, float v) {
  vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return v * mix(vec3(1.0), k, s);
}
void main() {
  vec3 base = uColorMode == 0
    ? hsv(0.62 - 0.55 * vShade, 0.75, 0.35 + 0.65 * vShade)
    : vCell;
  vec3 light = normalize(vec3(0.5, 0.8, 0.35));
  float diff = 0.45 + 0.55 * max(dot(vNormal, light), 0.0);
  outColor = vec4(base * diff, 1.0);
}`;

// 36 vertices of a unit cube centered on the origin: position then normal.
function cubeVertices() {
  const faces = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
  ];
  const out = [];
  for (const [n, u, v] of faces) {
    const corner = (su, sv) => [0, 1, 2].map((i) => 0.5 * n[i] + 0.5 * su * u[i] + 0.5 * sv * v[i]);
    const quad = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, -1), corner(1, 1), corner(-1, 1)];
    for (const p of quad) out.push(...p, ...n);
  }
  return new Float32Array(out);
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

const mat4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
  },
  lookAt(eye, target, up) {
    const sub = (a, b) => a.map((x, i) => x - b[i]);
    const norm = (a) => { const l = Math.hypot(...a); return a.map((x) => x / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = norm(sub(eye, target)), x = norm(cross(up, z)), y = cross(z, x);
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
  },
  mul(a, b) {
    const o = new Array(16).fill(0);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
    return o;
  },
};

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { antialias: true });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;
    this.canvas = canvas;
    this.count = 0;
    this.gridSize = 32;
    this.gap = 0.08;
    this.colorMode = 0;
    this.camera = { yaw: 0.7, pitch: 0.45, distance: 2.2 };

    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    this.prog = prog;
    this.u = Object.fromEntries(["uViewProj", "uSize", "uGap", "uColorMode"].map((n) => [n, gl.getUniformLocation(prog, n)]));

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const cube = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, cube);
    gl.bufferData(gl.ARRAY_BUFFER, cubeVertices(), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
    this.instances = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instances);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 16, 0);
    gl.vertexAttribDivisor(2, 1);
    gl.bindVertexArray(null);

    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    this.#bindControls();
  }

  /** Uploads packed [x, y, z, shade] cells. */
  setCells(data, gridSize) {
    const gl = this.gl;
    this.gridSize = gridSize;
    this.count = data.length / 4;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instances);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  }

  draw() {
    const { gl, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h);
    const bg = getComputedStyle(document.documentElement).getPropertyValue("--canvas-bg").trim() || "#0b0d12";
    const rgb = bg.match(/\w\w/g).map((x) => parseInt(x, 16) / 255);
    gl.clearColor(rgb[0], rgb[1], rgb[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const { yaw, pitch, distance } = this.camera;
    const r = distance * this.gridSize;
    const eye = [r * Math.cos(pitch) * Math.sin(yaw), r * Math.sin(pitch), r * Math.cos(pitch) * Math.cos(yaw)];
    const proj = mat4.perspective(Math.PI / 4, w / Math.max(h, 1), 0.1, r * 4);
    const viewProj = mat4.mul(proj, mat4.lookAt(eye, [0, 0, 0], [0, 1, 0]));

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.u.uViewProj, false, viewProj);
    gl.uniform1f(this.u.uSize, this.gridSize);
    gl.uniform1f(this.u.uGap, this.gap);
    gl.uniform1i(this.u.uColorMode, this.colorMode);
    gl.bindVertexArray(this.vao);
    if (this.count > 0) gl.drawArraysInstanced(gl.TRIANGLES, 0, 36, this.count);
    gl.bindVertexArray(null);
  }

  #bindControls() {
    const c = this.canvas, cam = this.camera;
    const pointers = new Map();
    let pinch = 0;
    c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]); });
    const end = (e) => { pointers.delete(e.pointerId); pinch = 0; };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("pointermove", (e) => {
      const prev = pointers.get(e.pointerId);
      if (!prev) return;
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch) this.#zoom(pinch / d);
        pinch = d;
        return;
      }
      cam.yaw -= (e.clientX - prev[0]) * 0.008;
      cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + (e.clientY - prev[1]) * 0.008));
    });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.#zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
  }

  #zoom(factor) {
    this.camera.distance = Math.max(0.6, Math.min(6, this.camera.distance * factor));
  }
}
