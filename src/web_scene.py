"""
Builds a self-contained Three.js scene: glowing additive-blended points for
the real connectome (foreground = auditory pathway subgraph, background =
whole-brain context silhouette), bloom post-processing, orbit camera, and
audio-synced pulsing (reads the actual <audio> element's currentTime each
frame and interpolates between precomputed activation snapshots -- this is
genuinely sample-accurate sync, unlike the earlier Plotly slider version).

All data is embedded inline as JSON in the returned HTML string so it can be
dropped straight into st.components.v1.html with no separate file server.
"""
import base64
import json
from pathlib import Path

import numpy as np

from brain_map_3d import load_positions, _to_scene_coords

SCALE = 1.0 / 60.0  # um -> scene units, keeps the whole structure in a compact range
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"


def _round(arr, decimals):
    return np.round(arr, decimals).tolist()


def build_scene_html(
    node_root_ids: np.ndarray,
    node_snapshots: list,
    is_seed: np.ndarray,
    audio_bytes: bytes,
    audio_mime: str,
    node_hop: np.ndarray = None,
    shell_style: str = "smooth",
) -> str:
    positions = load_positions()
    has_pos = np.array([rid in positions.index for rid in node_root_ids])
    kept_ids = node_root_ids[has_pos]
    pos_rows = positions.loc[kept_ids]
    fx, fy, fz = _to_scene_coords(pos_rows["x_nm"].to_numpy(), pos_rows["y_nm"].to_numpy(), pos_rows["z_nm"].to_numpy())

    # center everything around the foreground's centroid, then scale down
    cx, cy, cz = fx.mean(), fy.mean(), fz.mean()
    fx, fy, fz = (fx - cx) * SCALE, (fy - cy) * SCALE, (fz - cz) * SCALE

    fg_positions = np.stack([fx, fy, fz], axis=1)
    seed_kept = is_seed[has_pos]

    def transform_hull(verts_nm):
        v = np.array(verts_nm)
        hx, hy, hz = _to_scene_coords(v[:, 0], v[:, 1], v[:, 2])
        hx, hy, hz = (hx - cx) * SCALE, (hy - cy) * SCALE, (hz - cz) * SCALE
        return np.stack([hx, hy, hz], axis=1)

    with open(DATA_DIR / "hull_meshes.json") as f:
        all_hulls = json.load(f)
    hulls = all_hulls[shell_style]
    brain_hull_verts = transform_hull(hulls["brain"]["vertices"])
    vnc_hull_verts = transform_hull(hulls["vnc"]["vertices"])

    all_vals = np.concatenate([snap for _, snap in node_snapshots]) if node_snapshots else np.array([0.0, 1.0])
    # Anchor the dark end of the scale to true rest (V_REST=0), not a data-derived
    # percentile. Most neurons sit at exactly 0 (quiescent) most of the time -- if
    # cmin were allowed to go negative (pulled down by inhibited outliers), that
    # resting baseline would map to a visibly non-zero color, making the whole
    # population look "already lit" the instant any dynamics start.
    cmin = 0.0
    cmax = float(np.percentile(all_vals, 95))
    if cmax - cmin < 1e-3:
        cmax = 1.0

    # Color/brightness normalized PER HOP GROUP (seed / hop1 / hop2+), not against
    # one global scale. Seed neurons receive the raw external drive directly and
    # spike far more readily than downstream neurons that only get filtered
    # synaptic input -- against one global scale, seed sits pinned near the
    # brightest color almost the entire track regardless of the music (it always
    # dominates the global 95th percentile), which reads as "always lit at the
    # bottom" rather than responding to the track's actual dynamics. Normalizing
    # each hop group against its own range restores real quiet-vs-loud contrast
    # within the seed layer itself.
    hop_group = np.clip(node_hop[has_pos], 0, 2).astype(int) if node_hop is not None else np.zeros(len(fx), dtype=int)
    cmax_by_group = []
    for g in range(3):
        g_vals = np.concatenate([state[has_pos][hop_group == g] for _, state in node_snapshots]) if node_snapshots and (hop_group == g).any() else np.array([])
        g_cmax = float(np.percentile(g_vals, 95)) if g_vals.size else cmax
        cmax_by_group.append(g_cmax if g_cmax - cmin >= 1e-3 else cmax)

    times = [t for t, _ in node_snapshots]
    frames_values = [_round(state[has_pos], 3) for _, state in node_snapshots]

    data = {
        "fgPositions": _round(fg_positions.flatten(), 2),
        "isSeed": seed_kept.astype(int).tolist(),
        "hopGroup": hop_group.tolist(),
        "times": times,
        "frames": frames_values,
        "cmin": cmin,
        "cmax": cmax,
        "cmaxByGroup": cmax_by_group,
        "brainHull": {"vertices": _round(brain_hull_verts.flatten(), 2), "faces": hulls["brain"]["faces"]},
        "vncHull": {"vertices": _round(vnc_hull_verts.flatten(), 2), "faces": hulls["vnc"]["faces"]},
    }
    data_json = json.dumps(data, separators=(",", ":"))
    audio_b64 = base64.b64encode(audio_bytes).decode()

    return f"""
<div id="scene-container" style="width:100%; height:710px; display:flex; flex-direction:column-reverse; background:#03030a; border-radius:8px; overflow:hidden;">
  <div id="canvas-wrap" style="position:relative; flex:1 1 auto; min-height:0;">
    <div id="loading" style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#888; font-family:sans-serif; z-index:5;">Loading scene...</div>
    <div style="position:absolute; top:12px; left:12px; z-index:4; background:rgba(10,10,20,0.55); padding:10px 14px; border-radius:6px; font-family:sans-serif; font-size:12px; color:#bbb; max-width:260px; line-height:1.5;">
      <div style="color:#eee; font-weight:600; margin-bottom:4px;">Real FlyWire connectome</div>
      <div><span style="color:#5a7fd6;">&#9679;</span> blue shell = brain (from real neuron positions)</div>
      <div><span style="color:#8f5ad6;">&#9679;</span> purple shell = nerve cord (from real neuron positions)</div>
      <div><span style="color:#f0c527;">&#9679;</span> glowing points = auditory pathway (Johnston's Organ &#8594; AMMC &#8594; downstream), brightness/size = simulated activation</div>
      <div style="margin-top:6px; color:#888;">Drag to orbit &middot; scroll to zoom</div>
    </div>
  </div>
  <style>
    input[type=range].fly-slider {{
      -webkit-appearance: none;
      appearance: none;
      height: 4px;
      border-radius: 2px;
      background: linear-gradient(to right, #f0c527 var(--fill, 0%), #33333f var(--fill, 0%));
      outline: none;
      cursor: pointer;
      vertical-align: middle;
    }}
    input[type=range].fly-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 13px;
      height: 13px;
      border-radius: 50%;
      background: #fff;
      border: 2px solid #f0c527;
      box-shadow: 0 0 4px rgba(240,197,39,0.6);
      cursor: pointer;
      transition: box-shadow 0.15s ease;
    }}
    input[type=range].fly-slider:hover::-webkit-slider-thumb,
    input[type=range].fly-slider:active::-webkit-slider-thumb {{
      box-shadow: 0 0 8px rgba(240,197,39,0.9);
    }}
    input[type=range].fly-slider::-moz-range-track {{
      height: 4px;
      border-radius: 2px;
      background: #33333f;
    }}
    input[type=range].fly-slider::-moz-range-progress {{
      height: 4px;
      border-radius: 2px;
      background: #f0c527;
    }}
    input[type=range].fly-slider::-moz-range-thumb {{
      width: 13px;
      height: 13px;
      border-radius: 50%;
      background: #fff;
      border: 2px solid #f0c527;
      box-shadow: 0 0 4px rgba(240,197,39,0.6);
      cursor: pointer;
    }}
  </style>
  <div style="flex:0 0 auto; display:flex; align-items:center; gap:10px; padding:9px 14px; border-bottom:1px solid #1c1c26; background:#08080e; font-family:sans-serif;">
    <button id="playBtn" aria-label="Play" title="Play" style="background:#8f5ad6; color:#fff; border:none; width:34px; height:34px; border-radius:50%; cursor:pointer; display:flex; align-items:center; justify-content:center; box-shadow:0 0 10px rgba(143,90,214,0.5); transition:background 0.15s ease, box-shadow 0.15s ease;" onmouseover="this.style.background='#a170e0'" onmouseout="this.style.background='#8f5ad6'"><svg id="playIcon" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg></button>
    <input id="seek" class="fly-slider" type="range" min="0" max="1000" value="0" style="width:280px;">
    <span id="timeLabel" style="color:#ccc; font-size:12px; width:80px;">0:00 / 0:00</span>
    <label aria-label="Volume" title="Volume" style="color:#aaa; display:flex; align-items:center; gap:6px;">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M3 10v4h4l5 5V5L7 10H3z"/><path d="M16.5 12c0-1.77-.77-3.29-2-4.24v8.48c1.23-.95 2-2.47 2-4.24z"/></svg>
      <input id="vol" class="fly-slider" type="range" min="0" max="100" value="10" style="width:80px;">
    </label>
  </div>
  <audio id="player" style="display:none;">
    <source src="data:{audio_mime};base64,{audio_b64}" type="{audio_mime}">
  </audio>
</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from "three";
import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
import {{ EffectComposer }} from "three/addons/postprocessing/EffectComposer.js";
import {{ RenderPass }} from "three/addons/postprocessing/RenderPass.js";
import {{ UnrealBloomPass }} from "three/addons/postprocessing/UnrealBloomPass.js";

const DATA = {data_json};

function infernoColor(t) {{
  // hand-fit stops approximating the Inferno colormap
  const stops = [
    [0.0, [0,0,4]], [0.13,[40,11,84]], [0.29,[101,21,110]], [0.45,[159,42,99]],
    [0.6,[212,72,66]], [0.75,[245,125,21]], [0.88,[250,193,39]], [1.0,[252,255,164]]
  ];
  for (let i = 0; i < stops.length - 1; i++) {{
    const [t0, c0] = stops[i], [t1, c1] = stops[i+1];
    if (t >= t0 && t <= t1) {{
      const f = (t - t0) / (t1 - t0);
      return [
        (c0[0] + f*(c1[0]-c0[0]))/255,
        (c0[1] + f*(c1[1]-c0[1]))/255,
        (c0[2] + f*(c1[2]-c0[2]))/255,
      ];
    }}
  }}
  return [1,1,1];
}}

const container = document.getElementById('canvas-wrap');
const width = container.clientWidth, height = container.clientHeight;

// fit the camera to the actual structure instead of a fixed guess -- otherwise
// it can start zoomed into a dense sub-cluster and look like abstract noise
function computeBounds(flatArrays) {{
  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const arr of flatArrays) {{
    for (let i = 0; i < arr.length; i += 3) {{
      for (let d = 0; d < 3; d++) {{
        if (arr[i+d] < min[d]) min[d] = arr[i+d];
        if (arr[i+d] > max[d]) max[d] = arr[i+d];
      }}
    }}
  }}
  return {{ min, max, radius: Math.max(...max.map((v,i) => v-min[i])) / 2 }};
}}
function computeCentroid(arr) {{
  const sum = [0, 0, 0];
  const n = arr.length / 3;
  for (let i = 0; i < arr.length; i += 3) {{
    sum[0] += arr[i]; sum[1] += arr[i+1]; sum[2] += arr[i+2];
  }}
  return sum.map(v => v / n);
}}
const bounds = computeBounds([DATA.brainHull.vertices, DATA.vncHull.vertices, DATA.fgPositions]);
// Target the average position of the actual visible shell surfaces (brain +
// cord), not the bounding-box midpoint (which sits in empty space -- the cord
// extends asymmetrically) and not the sparse auditory-pathway neuron cluster
// alone (it's small and often dark/invisible before playback starts, and isn't
// positioned at the visual middle of the two shells). This is what's on screen
// most of the time, so it's what should be centered.
bounds.center = computeCentroid(DATA.brainHull.vertices.concat(DATA.vncHull.vertices));

// look slightly above the model's true center -- pushes the model itself
// lower within the frame, since OrbitControls always renders its target dead
// center
const lookY = bounds.center[1] + bounds.radius * 0.08;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, width/height, 0.01, 1000);
const camDist = bounds.radius * 0.85 * 0.9;  // 10% closer -> model reads ~10% larger on load
camera.position.set(bounds.center[0] + camDist*0.6, lookY + camDist*0.5, bounds.center[2] + camDist*0.6);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(width, height);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.insertBefore(renderer.domElement, container.firstChild);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(bounds.center[0], lookY, bounds.center[2]);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.minDistance = bounds.radius * 0.35;
controls.maxDistance = bounds.radius * 1.8;
controls.update();

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(width, height), 1.1, 0.5, 0.15);
composer.addPass(bloom);

// lighting for the solid hull shells
scene.add(new THREE.AmbientLight(0x404060, 1.2));
const keyLight = new THREE.DirectionalLight(0x8fb3ff, 1.4);
keyLight.position.set(4, 5, 3);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x6a3fa0, 0.8);
rimLight.position.set(-4, -2, -3);
scene.add(rimLight);

// real anatomical shape: convex hulls built directly from our own aligned
// neuron positions (brain + nerve cord, split by real region labels) --
// smooth, translucent, solid -- this is what makes it read as an actual brain
function buildHullMesh(hullData, color) {{
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(hullData.vertices, 3));
  geo.setIndex(hullData.faces.flat());
  geo.computeVertexNormals();
  const mat = new THREE.MeshPhysicalMaterial({{
    color, transparent: true, opacity: 0.16, roughness: 0.35, metalness: 0.05,
    side: THREE.DoubleSide, depthWrite: false, clearcoat: 0.3,
  }});
  return new THREE.Mesh(geo, mat);
}}
scene.add(buildHullMesh(DATA.brainHull, 0x5a7fd6));
scene.add(buildHullMesh(DATA.vncHull, 0x8f5ad6));

// foreground: the real auditory-pathway subgraph, custom shader for glow + per-point size/color
const nFg = DATA.fgPositions.length / 3;
const fgGeo = new THREE.BufferGeometry();
fgGeo.setAttribute('position', new THREE.Float32BufferAttribute(DATA.fgPositions, 3));
const colorAttr = new THREE.Float32BufferAttribute(new Float32Array(nFg * 3), 3);
const sizeAttr = new THREE.Float32BufferAttribute(new Float32Array(nFg), 1);
fgGeo.setAttribute('aColor', colorAttr);
fgGeo.setAttribute('aSize', sizeAttr);

const fgMat = new THREE.ShaderMaterial({{
  uniforms: {{ pixelRatio: {{ value: renderer.getPixelRatio() }} }},
  vertexShader: `
    attribute float aSize;
    attribute vec3 aColor;
    varying vec3 vColor;
    void main() {{
      vColor = aColor;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = aSize * (220.0 / -mvPosition.z);
      gl_Position = projectionMatrix * mvPosition;
    }}
  `,
  fragmentShader: `
    varying vec3 vColor;
    void main() {{
      float d = length(gl_PointCoord - vec2(0.5));
      float alpha = smoothstep(0.5, 0.0, d);
      gl_FragColor = vec4(vColor, alpha);
    }}
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
}});
scene.add(new THREE.Points(fgGeo, fgMat));

const baseSizes = new Float32Array(nFg);
for (let i = 0; i < nFg; i++) baseSizes[i] = DATA.isSeed[i] ? 0.09 : 0.05;

const spanByGroup = DATA.cmaxByGroup.map(cmax => Math.max(cmax - DATA.cmin, 1e-6));
// Per-group gamma, not one curve for everyone. Seed neurons receive raw drive
// directly and stay elevated most of the track even after per-group span
// normalization above (median activity sits well above half of their own
// peak) -- a >1 exponent compresses/darkens that group so genuinely quiet
// moments read dark instead of "always lit." Hop1/hop2+ are sparser (real LIF
// spiking is graded/patchy downstream), so they keep the original <1 boost
// that lifts moderate activity into visibly-lit range.
const gammaByGroup = [1.6, 1.0, 0.5];

function applyActivation(values) {{
  const colors = colorAttr.array, sizes = sizeAttr.array;
  for (let i = 0; i < nFg; i++) {{
    const group = DATA.hopGroup[i];
    const span = spanByGroup[group];
    const raw = Math.min(1, Math.max(0, (values[i] - DATA.cmin) / span));
    const norm = Math.pow(raw, gammaByGroup[group]);
    const [r,g,b] = infernoColor(norm);
    colors[i*3] = r; colors[i*3+1] = g; colors[i*3+2] = b;
    sizes[i] = baseSizes[i] * (1.0 + norm * 3.5);
  }}
  colorAttr.needsUpdate = true;
  sizeAttr.needsUpdate = true;
}}

function interpolatedValues(t) {{
  const times = DATA.times, frames = DATA.frames;
  if (times.length === 0) return new Array(nFg).fill(0);
  if (t <= times[0]) return frames[0];
  if (t >= times[times.length-1]) return frames[frames.length-1];
  let i = 0;
  while (i < times.length - 1 && times[i+1] < t) i++;
  const t0 = times[i], t1 = times[i+1];
  const f = (t - t0) / Math.max(t1 - t0, 1e-6);
  const a = frames[i], b = frames[i+1];
  const out = new Array(nFg);
  for (let k = 0; k < nFg; k++) out[k] = a[k] + f * (b[k] - a[k]);
  return out;
}}

applyActivation(new Array(nFg).fill(DATA.cmin));

// audio + transport controls
const player = document.getElementById('player');
const playBtn = document.getElementById('playBtn');
const seek = document.getElementById('seek');
const vol = document.getElementById('vol');
const timeLabel = document.getElementById('timeLabel');
player.volume = 0.1;

function fmt(s) {{ s = Math.floor(s || 0); return Math.floor(s/60) + ':' + String(s%60).padStart(2,'0'); }}

function updateFill(el) {{
  const pct = (el.value - el.min) / (el.max - el.min) * 100;
  el.style.setProperty('--fill', pct + '%');
}}
[seek, vol].forEach(updateFill);

const PLAY_ICON = '<path d="M8 5v14l11-7z"/>';
const PAUSE_ICON = '<path d="M6 5h4v14H6zM14 5h4v14h-4z"/>';
const playIcon = document.getElementById('playIcon');
function setPlayIcon(isPlaying) {{
  playIcon.innerHTML = isPlaying ? PAUSE_ICON : PLAY_ICON;
  playBtn.setAttribute('aria-label', isPlaying ? 'Pause' : 'Play');
  playBtn.setAttribute('title', isPlaying ? 'Pause' : 'Play');
}}

playBtn.onclick = () => {{
  if (player.paused) {{ player.play(); setPlayIcon(true); }}
  else {{ player.pause(); setPlayIcon(false); }}
}};
vol.oninput = () => {{ player.volume = vol.value / 100; updateFill(vol); }};
seek.oninput = () => {{
  if (player.duration) player.currentTime = (seek.value / 1000) * player.duration;
  updateFill(seek);
}};
player.load();
player.addEventListener('loadedmetadata', () => {{
  timeLabel.textContent = fmt(0) + ' / ' + fmt(player.duration);
}});
player.addEventListener('ended', () => {{ setPlayIcon(false); }});

let firstFrameRendered = false;
let currentDisplay = new Array(nFg).fill(DATA.cmin);  // true "off" state, not frame 0 of the precomputed track
const FADE = 0.93;  // when paused/stopped, activation fades toward off instead of freezing on the last frame
const FOLLOW = 0.12; // while playing, ease toward the target value each frame instead of snapping to it

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  if (player.duration) {{
    if (!player.paused && !player.ended) {{
      const target = interpolatedValues(player.currentTime);
      for (let i = 0; i < nFg; i++) currentDisplay[i] += (target[i] - currentDisplay[i]) * FOLLOW;
    }} else {{
      for (let i = 0; i < nFg; i++) currentDisplay[i] = DATA.cmin + (currentDisplay[i] - DATA.cmin) * FADE;
    }}
    applyActivation(currentDisplay);
    if (!seek.matches(':active')) {{ seek.value = (player.currentTime / player.duration) * 1000; updateFill(seek); }}
    timeLabel.textContent = fmt(player.currentTime) + ' / ' + fmt(player.duration);
  }}
  composer.render();
  if (!firstFrameRendered) {{
    firstFrameRendered = true;
    document.getElementById('loading').style.display = 'none';
  }}
}}
animate();

function resizeToContainer() {{
  const w = container.clientWidth;
  if (w < 10) return;
  camera.aspect = w / height;
  camera.updateProjectionMatrix();
  renderer.setSize(w, height);
  composer.setSize(w, height);
}}
window.addEventListener('resize', resizeToContainer);
// Streamlit's iframe often hasn't finished laying out to full width when this
// script first runs -- listening only to window 'resize' left the scene stuck
// with a wrong aspect ratio (model rendering off-center) until the user
// resized their browser. A ResizeObserver catches the real size once it settles.
new ResizeObserver(resizeToContainer).observe(container);
</script>
"""
