// Hero plate: the Mongol-Tori rocker-differential rover driving a procedural heightfield.
// ES module. Default export init(canvas, { glb, joints, poster, time }) returns a handle
// ({ ready, dispose, angles, stats, renderer }) or null when the scene must not start (no WebGL2).
// Frames: glTF Y-up, X forward, +Z rover right; metres. Joint pivots/axes come from rover-joints.json
// (every node frame is world-aligned at rest, so an axis is the same vector in node, parent and root).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DEG = Math.PI / 180;
const ROCKER_LIMIT = 15 * DEG;   // hard stops recommended by rocker-loop-sim (dead points at -21.5 / +40 deg)
const WHEEL_R = 0.15;
const SPEED = 0.32;              // m/s along the loop
const ORBIT_PERIOD = 70;         // s per camera revolution
const TERRAIN = 16, SEG = 128;   // 16 m x 16 m, 128 x 128 quads (0.125 m)
const LOOP_A = 2.5, LOOP_B = 1.75; // drive loop: ellipse semi-axes (x, z)
const FOV = 30;

// ---- terrain: seeded value noise (hashed lattice, smoothstep blend), 3 octaves, plus bumps and a trough
function hash(ix, iz) {
  let h = (ix * 374761393 + iz * 668265263 + 1013904223) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296 - 0.5;
}
function noise(x, z) {
  const ix = Math.floor(x), iz = Math.floor(z);
  let fx = x - ix, fz = z - iz;
  fx = fx * fx * (3 - 2 * fx); fz = fz * fz * (3 - 2 * fz);
  const a = hash(ix, iz), b = hash(ix + 1, iz), c = hash(ix, iz + 1), d = hash(ix + 1, iz + 1);
  return a + (b - a) * fx + (c - a) * fz + (a - b - c + d) * fx * fz;
}
function pathPoint(t, out) { return out.set(LOOP_A * Math.cos(t), 0, LOOP_B * Math.sin(t)); }
// bumps [x, z, radius, amplitude] sit on the loop, offset to one wheel track so the rockers differ
const BUMPS = [];
{
  const p = new THREE.Vector3();
  const spec = [[0.35, 1, 0.45, 0.26], [1.5, -1, 0.42, 0.23], [2.75, 1, 0.5, 0.28], [4.0, 0, 1.6, -0.14], [5.2, -1, 0.42, 0.25], [5.9, 1, 0.4, 0.2]];
  for (const [t, side, r, a] of spec) {
    pathPoint(t, p);
    const nx = LOOP_B * Math.cos(t), nz = LOOP_A * Math.sin(t), n = Math.hypot(nx, nz); // outward normal
    BUMPS.push([p.x + side * 0.41 * nx / n, p.z + side * 0.41 * nz / n, r, a]);
  }
}
function height(x, z) {
  let h = 0.11 * noise(x * 0.42 + 7.3, z * 0.42 + 3.1) + 0.05 * noise(x * 1.05 + 2.2, z * 1.05 + 9.7) + 0.016 * noise(x * 2.7 + 0.4, z * 2.7 + 5.5);
  for (const [bx, bz, r, a] of BUMPS) {
    const d2 = ((x - bx) * (x - bx) + (z - bz) * (z - bz)) / (r * r);
    if (d2 < 9) h += a * Math.exp(-d2);
  }
  return h;
}

// arc-length table so the rover moves at constant speed around the ellipse
const ARC_N = 720, ARC = new Float32Array(ARC_N + 1);
{
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  pathPoint(0, a);
  for (let i = 1; i <= ARC_N; i++) { pathPoint(i / ARC_N * 2 * Math.PI, b); ARC[i] = ARC[i - 1] + a.distanceTo(b); a.copy(b); }
}
const LOOP_LEN = ARC[ARC_N];
function paramAt(s) {
  s = ((s % LOOP_LEN) + LOOP_LEN) % LOOP_LEN;
  let lo = 0, hi = ARC_N;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ARC[m] <= s) lo = m; else hi = m; }
  return (lo + (s - ARC[lo]) / (ARC[hi] - ARC[lo])) / ARC_N * 2 * Math.PI;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const cssColor = (cs, name, fallback) => { const c = new THREE.Color(); try { c.setStyle(cs.getPropertyValue(name).trim() || fallback); } catch { c.setStyle(fallback); } return c; };

export default function init(canvas, opts = {}) {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let gl = null;
  try { gl = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'low-power' }); } catch { /* no WebGL */ }
  if (!gl) return null;
  const figure = canvas.closest('figure') || canvas.parentElement;

  const renderer = new THREE.WebGLRenderer({ canvas, context: gl, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.NoToneMapping; // keeps fog and background identical to the CSS plate colour

  const scene = new THREE.Scene();
  scene.background = new THREE.Color();
  scene.fog = new THREE.Fog(0xffffff, 5.5, 11);
  const camera = new THREE.PerspectiveCamera(FOV, 16 / 9, 0.1, 40);
  const hemi = new THREE.HemisphereLight(0xffffff, 0x808080, 0.9);
  const sun = new THREE.DirectionalLight(0xffffff, 2.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -2.4, right: 2.4, top: 2.4, bottom: -2.4, near: 1, far: 16 });
  sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
  scene.add(hemi, sun, sun.target);

  // terrain mesh, grid and the planned loop
  const geo = new THREE.PlaneGeometry(TERRAIN, TERRAIN, SEG, SEG).rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, height(pos.getX(i), pos.getZ(i)));
  geo.computeVertexNormals();
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
  const ground = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.95, metalness: 0 }));
  ground.receiveShadow = true;
  const gridPts = [];
  for (let k = 0; k <= TERRAIN * 2; k++) {
    const c = -TERRAIN / 2 + k * 0.5;
    for (let i = 0; i < SEG; i++) {
      const u0 = -TERRAIN / 2 + i * TERRAIN / SEG, u1 = u0 + TERRAIN / SEG;
      gridPts.push(u0, height(u0, c) + 0.004, c, u1, height(u1, c) + 0.004, c, c, height(c, u0) + 0.004, u0, c, height(c, u1) + 0.004, u1);
    }
  }
  const grid = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(gridPts, 3)),
    new THREE.LineBasicMaterial({ transparent: true, opacity: 0.14, depthWrite: false }));
  const loopPts = [];
  for (let i = 0, v = new THREE.Vector3(); i <= 400; i++) { pathPoint(i / 400 * 2 * Math.PI, v); loopPts.push(v.x, height(v.x, v.z) + 0.012, v.z); }
  const loop = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(loopPts, 3)),
    new THREE.LineDashedMaterial({ dashSize: 0.14, gapSize: 0.1, transparent: true, opacity: 0.7, depthWrite: false }));
  loop.computeLineDistances();
  scene.add(ground, grid, loop);

  // theme: plate, foreground and accent from the canvas's computed custom properties
  const ambientTop = new THREE.Color(), ambientBottom = new THREE.Color(), sand = new THREE.Color('#b9ab86'), dust = new THREE.Color('#8c8269');
  function retheme() {
    const cs = getComputedStyle(canvas);
    const plate = cssColor(cs, '--plate', cssColor(cs, '--bg', '#f5f6f1').getStyle());
    const fg = cssColor(cs, '--fg', '#1b1d19'), accent = cssColor(cs, '--accent', '#8a2b2b');
    const dark = (plate.r + plate.g + plate.b) / 3 < 0.2;
    scene.background.copy(plate); scene.fog.color.copy(plate);
    const hi = new THREE.Color().lerpColors(plate, sand, dark ? 0.5 : 0.55);
    const lo = new THREE.Color().lerpColors(plate, dust, dark ? 0.8 : 0.9);
    const col = geo.attributes.color, p = geo.attributes.position, c = new THREE.Color();
    for (let i = 0; i < p.count; i++) { c.lerpColors(lo, hi, clamp((p.getY(i) + 0.15) / 0.4, 0, 1)); col.setXYZ(i, c.r, c.g, c.b); }
    col.needsUpdate = true;
    grid.material.color.copy(fg); grid.material.opacity = dark ? 0.16 : 0.13;
    loop.material.color.copy(accent);
    hemi.color.copy(ambientTop.lerpColors(plate, new THREE.Color('#ffffff'), 0.4));
    hemi.groundColor.copy(ambientBottom.copy(lo));
    hemi.intensity = dark ? 0.7 : 0.95;
    sun.intensity = dark ? 1.9 : 2.1;
    sun.color.set(dark ? '#f2ead8' : '#fff6e6');
    if (!raf) renderOnce();
  }

  // rover
  const body = new THREE.Group();
  scene.add(body);
  const nodes = {}, missing = [];
  let J = null;
  const Y = new THREE.Vector3(0, 1, 0), _v = new THREE.Vector3(), _w = new THREE.Vector3(), _q = new THREE.Quaternion(), _sph = new THREE.Spherical();
  const axes = {}, rodEnd = {};
  const wheels = []; // [node, xBody, zBody, spin]
  let LF = 0.49, LB = 0.49, rodX = 0.389, endZ = 0.343, trackHalf = 0.414;

  const ready = (async () => {
    const [joints, gltf] = await Promise.all([
      fetch(opts.joints).then((r) => { if (!r.ok) throw new Error(`joints ${r.status}`); return r.json(); }),
      new Promise((res, rej) => {
        const draco = new DRACOLoader().setDecoderPath(new URL('../vendor/three/draco/', import.meta.url).href);
        new GLTFLoader().setDRACOLoader(draco).load(opts.glb, (g) => { draco.dispose(); res(g); }, undefined, rej);
      }),
    ]);
    J = joints;
    gltf.scene.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
    body.add(gltf.scene);
    for (const name of Object.keys(J)) {
      if (name.startsWith('_')) continue;
      const n = gltf.scene.getObjectByName(name);
      if (!n) { missing.push(name); continue; }
      nodes[name] = n;
      if (J[name].axis) axes[name] = new THREE.Vector3().fromArray(J[name].axis).normalize();
      if (J[name].end) rodEnd[name] = new THREE.Vector3().fromArray(J[name].end);
    }
    if (missing.length) console.warn('rover-scene: nodes missing from GLB, left static:', missing.join(', '));
    // geometry constants from the joints file (fall back to the CAD numbers above)
    if (J.fork_lf && J.fork_lb) { LF = J.fork_lf.pivot[0]; LB = -J.fork_lb.pivot[0]; }
    if (J.rod_left && J.rod_left.end) { rodX = -J.rod_left.pivot[0]; endZ = -J.rod_left.end[2]; }
    if (J._notes && J._notes.rest) trackHalf = J._notes.rest.track_m / 2;
    for (const [w, f, r] of [['wheel_lf', 'fork_lf', 'rocker_left'], ['wheel_lb', 'fork_lb', 'rocker_left'], ['wheel_rf', 'fork_rf', 'rocker_right'], ['wheel_rb', 'fork_rb', 'rocker_right']]) {
      if (!(nodes[w] && J[f] && J[r])) continue;
      wheels.push([nodes[w], J[r].pivot[0] + J[f].pivot[0] + J[w].pivot[0], J[r].pivot[2] + J[f].pivot[2] + J[w].pivot[2], 0]);
    }
    step(0);
    placeCamera();
    resize();
    figure && figure.classList.add('is-live');
    if (reduced) renderOnce(); else schedule();
  })();
  ready.catch((err) => { console.error('rover-scene: failed to start:', err); dispose(); });

  // ---- per-frame kinematics
  let s = SPEED * (Number(opts.time) || 0), lastYaw = null;
  const angles = { rockerL: 0, rockerR: 0, diffBar: 0, pitch: 0, roll: 0, yaw: 0, s: 0 };
  const _p = new THREE.Vector3(), _p2 = new THREE.Vector3();
  function step(dt) {
    s += SPEED * dt;
    const t = paramAt(s);
    pathPoint(t, _p);
    pathPoint(t + 1e-3, _p2).sub(_p);
    const yaw = Math.atan2(-_p2.z, _p2.x);
    const fx = Math.cos(yaw), fz = -Math.sin(yaw), rx = Math.sin(yaw), rz = Math.cos(yaw);
    const at = (xb, zb) => height(_p.x + fx * xb + rx * zb, _p.z + fz * xb + rz * zb);
    const hLF = at(LF, -trackHalf), hLB = at(-LB, -trackHalf), hRF = at(LF, trackHalf), hRB = at(-LB, trackHalf);
    const phiL = Math.atan2(hLF - hLB, LF + LB), phiR = Math.atan2(hRF - hRB, LF + LB);
    const pitch = (phiL + phiR) / 2;                      // body pitches by the rocker average
    const relL = clamp(phiL - pitch, -ROCKER_LIMIT, ROCKER_LIMIT), relR = clamp(phiR - pitch, -ROCKER_LIMIT, ROCKER_LIMIT);
    const roll = Math.asin(clamp(((hRF + hRB) - (hLF + hLB)) / 2 / (2 * trackHalf), -1, 1));
    // ponytail: diff bar from the left rod's vertical travel (asin), ~1:1 coupling; rod length error is < 3 mm at 15 deg.
    // Exact closure (circle/sphere intersection per rod) is the upgrade if the rods ever look stretched.
    const diffRoll = -Math.asin(clamp(rodX * Math.sin(relL) / endZ, -1, 1));
    body.position.set(_p.x, (hLF + hLB + hRF + hRB) / 4, _p.z);
    body.rotation.set(roll, yaw, pitch, 'YZX');
    const rot = (name, about) => { const n = nodes[name], a = axes[name]; if (n && a) n.quaternion.setFromAxisAngle(a, about); };
    rot('rocker_left', relL * Math.sign(axes.rocker_left ? axes.rocker_left.z : 1));
    rot('rocker_right', relR * Math.sign(axes.rocker_right ? axes.rocker_right.z : 1));
    rot('diff_bar', diffRoll * Math.sign(axes.diff_bar ? axes.diff_bar.x : 1));
    // wheels spin by distance; the outer side runs faster through a turn
    const dyaw = lastYaw === null ? 0 : Math.atan2(Math.sin(yaw - lastYaw), Math.cos(yaw - lastYaw));
    lastYaw = yaw;
    for (const w of wheels) {
      w[3] += (SPEED * dt + dyaw * w[2]) / WHEEL_R;
      const a = axes[w[0].name];
      if (a) w[0].quaternion.setFromAxisAngle(a, -w[3] * Math.sign(a.z));
    }
    body.updateMatrixWorld(true);
    // rods: aim each from its lower ball (on the rocker) to its upper ball (on the diff bar)
    for (const name of ['rod_left', 'rod_right']) {
      const rod = nodes[name], bar = nodes.diff_bar, end = rodEnd[name];
      if (!(rod && bar && end)) continue;
      bar.localToWorld(_w.copy(end)).sub(rod.getWorldPosition(_v));
      rod.parent.getWorldQuaternion(_q).invert();
      rod.quaternion.setFromUnitVectors(Y, _w.applyQuaternion(_q).normalize());
    }
    // never let a wheel sink below the terrain: lift the body by the worst penetration
    let lift = 0;
    for (const w of wheels) { w[0].getWorldPosition(_v); lift = Math.max(lift, height(_v.x, _v.z) + WHEEL_R - _v.y); }
    if (lift > 0) { body.position.y += lift; body.updateMatrixWorld(true); }
    sun.target.position.copy(body.position);
    sun.position.copy(body.position).add(_v.set(3.5, 6.5, 2.5));
    Object.assign(angles, { rockerL: relL / DEG, rockerR: relR / DEG, diffBar: diffRoll / DEG, pitch: pitch / DEG, roll: roll / DEG, yaw: yaw / DEG, s });
  }

  // ---- camera: orbit around the rover, auto-rotating until the visitor takes over
  const controls = new OrbitControls(camera, canvas);
  Object.assign(controls, { enablePan: false, enableDamping: !reduced, dampingFactor: 0.08, minDistance: 2.6, maxDistance: 8, minPolarAngle: 0.35, maxPolarAngle: 1.5, autoRotate: !reduced, autoRotateSpeed: 60 / ORBIT_PERIOD, zoomSpeed: 0.6 });
  const stopAuto = () => { controls.autoRotate = false; };
  controls.addEventListener('start', stopAuto);
  controls.addEventListener('change', () => { if (!raf) renderOnce(); });
  const target = new THREE.Vector3();
  function placeCamera() {
    target.copy(body.position).y += 0.5;
    controls.target.copy(target);
    const dist = 4.2 / Math.min(1, camera.aspect / 1.3);
    // open on a 3/4 view from the rover's front-left, 20 deg up, whatever its heading
    camera.position.copy(target).add(_v.set(0.72, 0.34, -0.6).normalize().multiplyScalar(dist).applyAxisAngle(Y, body.rotation.y));
    controls.update();
  }
  function followRover() {
    _v.copy(body.position); _v.y += 0.5;
    target.lerp(_v, 0.08);
    _w.copy(target).sub(controls.target);
    camera.position.add(_w);
    controls.target.copy(target);
  }
  const KEYS = { ArrowLeft: [5 * DEG, 0], ArrowRight: [-5 * DEG, 0], ArrowUp: [0, -3 * DEG], ArrowDown: [0, 3 * DEG] };
  function onKey(e) {
    const k = KEYS[e.key];
    if (!k || e.altKey || e.ctrlKey || e.metaKey) return;
    e.preventDefault();
    stopAuto();
    _sph.setFromVector3(_v.copy(camera.position).sub(controls.target));
    _sph.theta += k[0];
    _sph.phi = clamp(_sph.phi + k[1], controls.minPolarAngle, controls.maxPolarAngle);
    camera.position.copy(controls.target).add(_v.setFromSpherical(_sph));
    controls.update();
    if (!raf) renderOnce();
  }
  canvas.addEventListener('keydown', onKey);

  // ---- lifecycle
  const stats = { frames: [], nodes: () => Object.keys(nodes), missing };
  let raf = 0, lastT = 0, visible = true, disposed = false;
  function frame(now) {
    const t0 = performance.now();
    const dt = lastT ? Math.min((now - lastT) / 1000, 0.1) : 0;
    lastT = now;
    step(dt);
    followRover();
    controls.update(dt);
    renderer.render(scene, camera);
    stats.frames.push(performance.now() - t0);
    if (stats.frames.length > 240) stats.frames.shift();
    raf = 0;
    schedule();
  }
  // one frame on demand (reduced motion, theme change, resize while paused); no controls.update() here, it
  // would fire 'change' and recurse
  function renderOnce() { if (!disposed && J) { followRover(); renderer.render(scene, camera); } }
  function schedule() {
    const want = !disposed && J && !reduced && visible && !document.hidden;
    if (want && !raf) raf = requestAnimationFrame(frame);
    if (!want && raf) { cancelAnimationFrame(raf); raf = 0; lastT = 0; }
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (!raf) renderOnce();
  }
  const io = new IntersectionObserver((en) => { visible = en[0].isIntersecting; schedule(); });
  io.observe(canvas);
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  const scheme = matchMedia('(prefers-color-scheme: dark)');
  scheme.addEventListener('change', retheme);
  const mo = new MutationObserver(retheme);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
  document.addEventListener('visibilitychange', schedule);
  addEventListener('pagehide', dispose);
  retheme();

  function dispose() {
    if (disposed) return;
    disposed = true;
    schedule();
    io.disconnect(); ro.disconnect(); mo.disconnect();
    scheme.removeEventListener('change', retheme);
    document.removeEventListener('visibilitychange', schedule);
    removeEventListener('pagehide', dispose);
    canvas.removeEventListener('keydown', onKey);
    controls.dispose();
    scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    renderer.dispose();
    figure && figure.classList.remove('is-live');
  }

  return { ready, dispose, angles: () => ({ ...angles }), stats, renderer, scene, camera, reduced };
}
