# Build Story — fly-music

Simulating fly auditory-pathway activation in response to rock/rap/classical music, using real FlyWire connectome data as the underlying graph.

---

## 2026-08-16 — Kickoff

**Goal:** use FlyWire's Drosophila connectome to see if we can plausibly "mimic" fly behavior under different music genres. Started from a specific FlyWire neuron page (`local_id=6319f98e85561543a855e5490ec9cb61`), which turned out to be a JS-rendered neuron viewer — not directly scrapable, so we pivoted to using FlyWire's real programmatic tools instead of that one page.

**Reality check up front:** FlyWire gives a real, detailed *wiring diagram* (which neurons connect to which, how strongly). It does not give behavior data or any "reaction to music" dataset — there's no ground truth to validate against. So this project is a **connectome-inspired simulation / art piece**, not a validated behavior predictor. Framing that stays explicit throughout.

**Research findings:**
- Programmatic access is via `caveclient` (CAVE = Connectome Annotation Versioning Engine), `pip install caveclient`.
- The real auditory pathway exists and is mapped: Johnston's Organ (~400 mechanosensory neurons in the antenna, detects sound/wind/gravity) → AMMC (antennal mechanosensory motor center) → downstream. A 2024 whole-brain connectome paper traced 178 neurons in this pathway — that's our seed set.
- Codex (codex.flywire.ai) has a UI for browsing cell-type annotations, including dedicated auditory-cell labels, and a "Download Data" app for CSV exports as an alternative to the API.

**Stack chosen:**
- `caveclient` — pull real connectome data
- `networkx` — hold it as a weighted directed graph
- `librosa` — extract audio features (tempo, amplitude envelope, frequency bands) from tracks
- `numpy` / `scipy` — leaky-integrate-and-fire style propagation model driven by audio features
- `matplotlib` — first-pass visualization

**Scaffolded project** at `~/fly-music` (later moved, see below):
```
fly-music/
  data/{connectome,audio}/
  src/
  notebooks/
  outputs/
  requirements.txt
  README.md
```
Installed deps into a venv. Wrote `src/check_auth.py` as a minimal "does auth work" smoke test before touching real data.

**Moved project → `/Users/jdon/Desktop/fly`** per request. Rebuilt the venv from scratch at the new path (venvs bake in absolute paths in their activate scripts, so a plain `mv` would've left it broken).

**Auth:**
- Generated a CAVE token via `client.auth.get_new_token()`, approved in the browser already signed into FlyWire, saved locally with `client.auth.save_token()` (lives in `~/.cloudvolume/secrets/`, outside the project folder — not something that'll end up committed anywhere).
- Ran `check_auth.py` → token itself is valid (passed authentication), but hit a **403: missing "view" permission for dataset fafb**. Being signed into FlyWire's web UI does not automatically grant CAVE API dataset access — that's a separate grant.

**Current blocker:** need FAFB dataset view permission. Two options being tried:
1. Codex's own "Download Data" app in-browser (may not gate on the same CAVE permission).
2. Request dataset access directly (form on flywire.ai/data, or email flywire@princeton.edu) — manual step, waiting on user.

**Next:** resolve data access, then write `src/fetch_connectome.py` to pull the Johnston's Organ → AMMC → downstream neuron set and their synaptic weights.

---

## 2026-08-16 (cont.) — Data access resolved via manual download

Instead of fighting CAVEclient's separate dataset-permission gate, downloaded the public data exports directly from Codex/FlyWire's Download Data app:
- `neurons.csv.gz` — 158,262 neurons, female brain + nerve cord dataset (matches Codex's published "BANC v888" release). Columns include Root ID, Class/Sub Class, Nerve, Primary Cell Type, etc.
- `connections_princeton.csv.gz` — 3,990,040 synaptic connections (`pre_root_id, post_root_id, neuropil, syn_count, nt_type`).

Both moved into `data/connectome/`.

**Confirmed the auditory pathway is present in this data:**
- 4,502 neurons tagged with `left_antennal_nerve` / `right_antennal_nerve` (afferent, sensory) — this is where Johnston's Organ mechanosensory neurons project through.
- `AMMC_L` / `AMMC_R` (antennal mechanosensory motor center) appear as neuropil labels in the connections table — the confirmed downstream target of Johnston's Organ.

This gives us everything needed to build the seed neuron set → downstream propagation graph without needing the CAVE API permission at all.

**Publication/citation obligations (must follow if this project is ever shared publicly, not just kept local):**
- Data is under **CC BY-NC 4.0** — non-commercial, attribution required.
- Cite Codex: `http://dx.doi.org/10.13140/RG.2.2.35928.67844`, plus the underlying method papers listed in FlyWire's citation table (synapse detection, neurotransmitter prediction, etc. — see [codex.flywire.ai/about_flywire](https://codex.flywire.ai/about_flywire)).
- Credit Princeton (Brain Initiative grants MH117815, MH129268, U24 NS126935) and the FlyWire proofreading community.
- This applies to **published/released reconstructions**, which the public CSV downloads are — no separate per-lab consent needed since we're using the already-published data release, not unpublished production data. (Unpublished/production-only neurons would trigger the stricter pre-publication consent process in [edit.flywire.ai/principles.html](https://edit.flywire.ai/principles.html); doesn't apply here.)
- Action item: when/if this project is shown publicly, add a citation block to the README before sharing.

**Next:** write `src/fetch_connectome.py` to filter `neurons.csv.gz` for the antennal-nerve seed set, pull their downstream connections (via AMMC and beyond) from `connections_princeton.csv.gz`, and build the `networkx` graph.

---

## 2026-08-16 (cont.) — First real graph built

Wrote `src/fetch_connectome.py`:
- Seed set = 4,502 neurons on `left_antennal_nerve` / `right_antennal_nerve` (Johnston's Organ afferents).
- BFS outward 2 hops through real synaptic connections, keeping only edges with `syn_count >= 5` (the raw connectome is dense — without a threshold, expansion balloons fast).
- Saved as a `networkx` DiGraph (pickled) plus a CSV of node metadata (cell class/type/nerve per neuron).

**Result:** 25,971 neurons, 122,317 edges. Ran in ~3 seconds — the CSVs are small enough that no database or heavier tooling is needed for this project's scale.

**Note for later:** 2 hops from the sensory periphery already reaches ~16% of the whole brain (25,971 of ~158,262 total neurons) — broader than ideal for a clean "auditory pathway" story. `N_HOPS` and `MIN_SYN` are exposed as constants at the top of the script so this can be tightened (fewer hops, higher synapse threshold, or filtering to neurons that pass through `AMMC_L`/`AMMC_R` specifically) once we see whether the simulation runs cleanly at this size, or whether it needs to be pruned first.

**Next:** `src/audio_features.py` — extract tempo/amplitude/frequency-band features from a rock/rap/classical track each, to use as the input drive for the graph simulation.

---

## 2026-08-16 (cont.) — Model choice + genre set finalized

**Model:** staying on Sonnet for the build. The genre/audio work doesn't route through an LLM at all — librosa extracts tempo/amplitude/frequency features numerically from the audio file directly — so model choice barely affects this project's core logic.

**Genre set locked in:** rock, noise, classical (dropped rap). Tracks:
- Rock: "Comfortably Numb" (Pink Floyd)
- Classical: a Chopin Nocturne recording
- Noise: generated synthetically (not a copyrighted recording, so no licensing concern) — used as a "no musical structure" control/baseline to compare the two real tracks against.

**Licensing note:** Chopin's Nocturnes are public-domain compositions, but the specific *recording* used is still typically copyrighted to the performer/label; Comfortably Numb is fully copyrighted. Personal, local feature-extraction analysis of legally-owned audio is normal fair-use territory — the constraint is not to publish/redistribute the audio files themselves anywhere public (repo, artifact, etc.). Only extracted numeric features and resulting visualizations should ever leave the machine.

Generated `data/audio/noise_white.wav` and `noise_pink.wav` (30s, 44.1kHz) via numpy/scipy — pink noise will be used as the actual "noise" condition since its energy distribution is closer to real music than flat white noise, making it a fairer comparison baseline.

**Next:** waiting on the two real audio files to be added to `data/audio/`, then `src/audio_features.py`.

---

## 2026-08-16 (cont.) — Pivoted to drag-and-drop, no stored audio files

No legal copies of Comfortably Numb / Chopin Nocturne on hand. Rather than source them, pivoted the whole design: instead of a fixed rock/noise/classical dataset baked into the repo, built an **interactive app where you drag in any audio file at runtime**. This is a better solution on two fronts:
- Sidesteps licensing entirely — audio is read into memory, processed, and discarded; nothing is ever saved to disk or published.
- More useful as a demo/portfolio piece than three static precomputed results — you can react to any track live.

**Built the full pipeline:**
- `src/audio_features.py` — extracts tempo, RMS envelope, onset strength, and bass/mid/treble band energy from an in-memory audio array via `librosa`, resampled to a fixed 150 frames regardless of track length (so any song drives the sim the same way).
- `src/simulate.py` — loads the 25,971-node auditory graph, builds a sparse weighted adjacency matrix (edge weight = log-scaled synapse count, normalized 0-1), computes hop-distance from the seed layer via multi-source BFS, then runs a simple recurrent update per audio frame: `state = tanh(decay*state + drive*seed_mask + alpha*incoming)`. Explicitly documented as a speculative graph-propagation toy, not a validated model of real neural dynamics.
- `src/app.py` — Streamlit app: drag-and-drop file uploader (or pick the white/pink noise control), runs the pipeline, shows a live activation chart (seed / hop1 / hop2+ over time) and a table of the top 15 most-activated neurons with real cell-type labels and root IDs.

**Dependencies added:** `streamlit`, `plotly` (installed but not yet used, reserved for a richer graph view later), `ffmpeg` (via Homebrew — needed for `librosa` to decode MP3s, not just WAV).

**Smoke-tested** end-to-end with the pink noise control — pipeline runs in well under a second for a 30s track (26k-node sparse matrix-vector propagation × 150 frames is cheap).

**Launched locally:** `streamlit run src/app.py --server.address localhost` → http://localhost:8501. Deliberately bound to localhost only (Streamlit defaults to binding all network interfaces, which isn't needed for a local personal tool).

**Noted for later:** user has access to additional FlyWire datasets beyond FAFB/BANC — notably **MANC** (male nerve cord, 23,665 neurons), which is arguably more relevant for "behavior" than brain-only data, since motor output originates in the nerve cord. Worth expanding the graph to include nerve cord connectivity once the brain-only version is validated as working well.

**Next:** try it live with real tracks via drag-and-drop (no data pipeline changes needed), see if the activation patterns look meaningfully different across genres, then decide whether to tune `N_HOPS`/`MIN_SYN` (current graph reaches ~16% of the brain, broader than ideal) or add a live-streaming chart update instead of compute-then-display.

---

## 2026-08-16 (cont.) — Brain region visualization for portfolio use

Wanted a visually pleasing "which part of the brain fires" view, not just line charts. No per-neuron x/y/z coordinates exist in the public CSV export (that needs mesh access via the CAVE API, which we're still locked out of on dataset permissions) — but the `Top in/out region` field on every neuron gives a real named anatomical region (e.g. `AMMC`, `AL`, `MB_CA`, `T1_PRONM`), and this BANC dataset already includes nerve cord segments, so the earlier "more brains/MANC" idea is already covered without a separate pull.

- Added `region` attribute to `fetch_connectome.py`'s node metadata (primary token from `Top in/out region`, compound labels like `ME.LO` collapsed to `ME`). Re-ran the fetch — same 25,971/122,317 node/edge counts, now with region labels attached.
- Extended `simulate.py` to track **per-region, per-frame activation** (not just per-hop-layer), using `np.bincount` for a cheap weighted mean per timestep. Filters out regions with fewer than 15 neurons in the subgraph to avoid noisy single-neuron means.
- Built `src/brain_map.py`: a hand-authored schematic layout of ~40 real fly neuropils (optic lobes flanking center, central brain ring, gnathal ganglion at the base, nerve cord segments chained below — topologically faithful to real fly neuroanatomy, but not pulled from an actual 3D mesh, so explicitly labeled "schematic, not to scale" in the app). Renders via Plotly: bubble size = neuron count, color = activation (Inferno scale), dark theme, hover tooltips with real region names and counts.
- Wired into `app.py` as a new section below the activation-over-time chart.

**Verified with pink noise control:** 41 regions surfaced, AMMC (the real auditory center) shows high activation as expected given it's directly downstream of the seed layer — a reasonable sanity check that the pipeline is behaving sensibly before testing real music.

**Next:** test with real tracks dragged into the running app, see how region-level patterns differ by genre/track character.

---

## 2026-08-16 (cont.) — Fixed saturation bug, added pulsing animation

User feedback after trying it live: activation looked "too uniform" across tracks.

**Root cause found:** the recurrent propagation term summed *raw* incoming synaptic weight per neuron — some neurons have incoming-strength up to 242 in this graph. `alpha * incoming` (alpha=0.6) blew straight through `tanh`'s saturation ceiling within a couple of frames and pinned activation near 1.0 for most of the graph regardless of what was playing. Fixed in `simulate.py` by column-normalizing the adjacency matrix so `incoming` is a weighted *average* of predecessor activation (bounded ~0-1) instead of an unbounded sum. Verified with a synthetic pulsing test signal: activation now genuinely rises and falls (0.44-0.82 range) tracking the input instead of flatlining at ceiling.

**Added pulsing animation:** `brain_map.py` now has `build_animated_figure()` — Plotly frames (subsampled every 3rd of 150 frames = 51 steps) with a Play/Pause button and scrubber, so the schematic bubbles actually change color over the course of the track instead of showing only a final snapshot. `app.py` now also plays the audio (`st.audio`) alongside so you can listen while watching it pulse (not sample-accurate synced playback, but close enough to follow along).

**Next:** get real tracks tested live; check whether region-level pulse patterns meaningfully differ by genre now that the saturation bug is fixed.

---

## 2026-08-16 (cont.) — Found real 3D coordinates, unblocking true 3D visualization

User asked how to get CAVE dataset permission properly (rather than working around it). Turned out the earlier 403 wasn't a real permission block — it was the wrong datastack name. We were requesting `flywire_fafb_production` (the live, contributor-only editing dataset). The **published, publicly-accessible** datastack for our downloaded data is `brain_and_nerve_cord_public` (BANC) — confirmed matching version (888) to what we already have locally. No special access request needed; our existing token already works against it.

This unlocks real per-neuron 3D coordinates via the `cell_representative_point` CAVE table (`pt_root_id`, `pt_position` in nm). Checked coverage against our 25,971-neuron subgraph: **99.2%** of neurons have a representative point (2,578 of 2,598 in a test batch). This is much lighter than pulling full neuron meshes (no `cloudvolume`/mesh streaming needed) and gives genuine anatomical positions rather than the hand-placed schematic layout.

**Next:** pull `pt_position` for the full subgraph via `cell_representative_point`, cache locally (avoid re-querying CAVE every run), and build a real 3D scatter view (Plotly `Scatter3d`) driven by the same per-frame region/neuron activation already computed in `simulate.py`. The 2D schematic (`build_figure`/`build_animated_figure`) stays as a fast fallback/comparison view.

---

## 2026-08-16 (cont.) — Real 3D visualization shipped

Wrote `src/fetch_coordinates.py`: queries `cell_representative_point` on `brain_and_nerve_cord_public` for every neuron in our subgraph, batched (500/request), cached to `data/connectome/neuron_positions.csv`. Coverage: **25,760 / 25,971 neurons (99.2%)** — essentially the whole subgraph has a real anatomical position.

Extended `simulate.py` with an optional `node_snapshot_stride` param: instead of only tracking aggregate layer/region means, it now optionally saves full per-neuron activation snapshots at intervals (every 10th frame -> 16 snapshots for a 150-frame run) as float32 arrays, keeping memory reasonable (~1.6MB total, not 150 full copies).

Built `src/brain_map_3d.py`: aligns node positions to activation snapshots by root_id, renders a Plotly `Scatter3d` (nm converted to um, x-flipped to read the same direction as the schematic) with the same Play/Pause/scrubber animation pattern as the 2D version. Seed-layer neurons rendered slightly larger for visual anchoring.

Wired into `app.py` as the **default view**, with a radio toggle to fall back to the faster 2D schematic. Performance: full simulation + 3D figure build together take well under a second locally (25,760 real points, 16 frames).

This replaces the hand-placed schematic coordinates with genuine anatomical positions — a real visual portfolio piece: an actual fly brain + nerve cord, built from real FlyWire/CAVE data, pulsing in response to whatever audio is dragged in.

**Next:** try it live, check whether the 3D view is legible/performant in-browser (25k points is a lot for WebGL scatter -- may need point-count reduction or opacity tuning if it feels cluttered), and get feedback on whether genre differences are now visible.

---

## 2026-08-16 (cont.) — Fixed uniform/invisible brain feedback, raised the visual bar

User feedback on the first Plotly 3D pass: "can't really see a brain" (background silhouette wasn't visible, colors washed out, geometry squashed) and later "nothing fires up when the track plays." Root causes and fixes:
- Nerve cord axis (z) is ~8x longer than the brain axis (x) in real coordinates -- `aspectmode="data"` was crushing everything into a thin sliver. Switched to `aspectmode="cube"`.
- Background silhouette (`context_positions.csv`, ~19.6k random neurons across the whole brain+cord, fetched via a new `fetch_background_positions.py`) was too small/faint to read -- bumped size, opacity, color.
- Color contrast was fixed at 0-1 regardless of the actual activation range each run produced -- switched to auto-scaling contrast (2nd/98th percentile of that run's values).
- Pulsing was color-only, too subtle to read as "firing" -- added size pulsing too (activated neurons visibly grow, not just recolor).
- Removed a stray "trace 1" legend label overlapping the colorbar.

**Bigger ask:** user referenced a known project where someone rendered the FlyWire connectome as a Minecraft world with a fly flying through it, and wanted matching visual calibre. Assessed the two paths: a literal Minecraft build would only support a static one-time colored snapshot (block-update throughput can't support live per-frame pulsing across tens of thousands of blocks), which would mean giving up the live audio-reactive interactivity already built. Recommended a Three.js glow scene instead -- same visual register (striking, glowing, portfolio-grade) but keeps everything live. User chose Three.js.

**Built `src/web_scene.py`:** a self-contained Three.js scene (additive-blended custom shader points, bloom post-processing via `UnrealBloomPass`, `OrbitControls` with auto-rotate-when-idle) embedded directly into the Streamlit app via `components.html` -- no separate file server needed, all data (positions, per-frame activation snapshots, audio as base64) inlined as JSON in the returned HTML string.

**Real improvement over the Plotly version, not just prettier:** pulsing is now driven by the actual `<audio>` element's `currentTime` read every animation frame and interpolated between precomputed snapshots -- genuinely synced to playback position, not a slider approximating it. Custom playback UI (play/pause, seek, volume) built directly into the scene.

Wired in as the new default view in `app.py`, with the Plotly 3D and 2D schematic views kept as fallback toggles. Bumped Streamlit's `--server.maxMessageSize` to 500 (payload is ~15MB for a 30s test track -- mostly audio + activation data -- comfortably local/no-network, but worth the headroom for longer real songs).

**Next:** test live with a real track, check WebGL performance/framerate is smooth, verify the glow/bloom aesthetic actually reads well, and confirm pulsing is now clearly visible against the background silhouette.

---

## 2026-08-16 (cont.) — Fixed low-poly hulls, camera framing, legend, and a genuine geometry bug

User feedback: "lighting/glowing is nice but the background seems like random vectors/shapes" and "not really interactive." Root causes:
- Raw `ConvexHull` output is very low-poly (~76-120 vertices) -- looked like a faceted crystal, not an organic brain shape. Fixed by subdividing + Laplacian-smoothing the hull mesh (`trimesh`) before export -- brain hull now ~4,700 verts. First attempt used `volume_constraint=True` in the smoother, which produced NaN vertices on the brain hull (negative-volume edge case) -- disabled that flag once confirmed the unconstrained version has zero NaNs.
- No camera framing or on-screen legend -- looked like abstract art with no indication of what was being shown. Added a `computeBounds()` pass in JS that fits the camera to the actual structure on load, and a legend overlay explaining what each color/shape means plus interaction hints.

**Follow-up bug, caught by user screenshot ("what is this shape?"):** the nerve cord rendered as a thin blade/needle, not a blob. Real cause: a single convex hull cannot represent a curving structure -- it just draws the tightest convex wrap, cutting straight across the cord's natural curve and collapsing it into a wedge. This is a geometry limitation, not a smoothing bug. Fixed by splitting the VNC point cloud into 6 segments along its own principal axis (via SVD, not string-matching region labels -- more robust given some VNC sub-region labels have very few points) and hulling each segment separately, producing a chain of blobs that follows the real curve. Colors ramp along the chain (HSL gradient) so it still reads as one continuous structure.

**Next:** get live confirmation the shape reads correctly now (brain blob + curving segmented cord, not a blade), and that the legend/framing fixes make it feel like actual data rather than abstract art.

---

## 2026-08-16 (cont.) — Found the real bug behind the blade shape

Segmenting the nerve cord (previous entry) didn't fix it -- screenshots still showed a single blue blade, no visible purple chain. Investigated directly rather than guessing further: printed the actual brain hull bounding box and it was `[1.15M, 2.6M, 7.9M]` -- the z-axis was still body-length, meaning the "brain" bucket itself was contaminated.

Root cause: 1,197 neurons in the context sample are labeled `NO_CONS` ("no consensus" -- FlyWire couldn't confidently resolve their region) with z-coordinates up to 11.8M, i.e. scattered anywhere in the body including deep in the nerve cord. Since `"NO_CONS"` doesn't match any of the VNC marker strings, these fragments fell into the "brain" bucket by default via `~merged["is_vnc"]`, dragging the brain hull's bounding box the full length of the body.

Fix: exclude any neuron without a real resolved region label (`""` or `"NO_CONS"`) from the hull classification entirely, for both brain and VNC. Verified after the fix: brain bbox is now `[885k, 835k, 283k]` -- proportional, no runaway axis -- and the 6 VNC segment centers progress smoothly along the body axis (y: 329k -> 768k), confirming a coherent chain rather than a degenerate shape.

**Lesson for this project:** always sanity-check bounding boxes/extents directly when a shape looks wrong, rather than iterating on the smoothing/rendering code -- the actual bug was three edits back in the region-classification step, not in geometry processing at all.

**Next:** confirm live that the brain now reads as a rounded blob and the nerve cord as a proper curving chain, not a blade.

---

## 2026-08-16 (cont.) — Switched convex hull to density-based isosurface

User asked directly: "does this make sense scientifically?" Honest answer was no -- a convex hull can only produce a *convex* shape, mathematically incapable of representing the concavities of a real fly brain (the cleft between optic lobes and central brain) or the nerve cord's curve. It was a geometric artifact, not anatomy, however smoothed.

Rebuilt `fetch_hull_meshes.py` around a proper method: voxelize the real neuron point cloud into a 3D occupancy grid, Gaussian-smooth it, then run marching cubes (`skimage.measure`) to extract a density isosurface. This can represent concavity, so it actually follows the point cloud's local shape rather than wrapping only the outermost extreme points. Level threshold chosen so ~85% of real points fall inside the surface.

This also let us **drop the whole 6-segment nerve-cord chain workaround** from the previous fix -- a density isosurface naturally follows a curving point cloud without needing to be artificially split into pieces, since (unlike convex hull) it isn't a single global wrap.

Filtered each surface to its single largest connected component (density noise was spawning a handful of tiny stray specks). Result: brain surface 3,878 verts/7,752 faces, nerve cord 2,828 verts/5,652 faces. Sanity-checked the brain's proportions against the independent `flybrains` BANC template mesh fetched earlier (different coordinate convention, so not directly overlaid, but bbox aspect ratio -- one long axis, two shorter comparable axes -- matches, which is a reasonable independent cross-check).

**Next:** confirm live that this reads as an actual lobed/rounded brain shape rather than a wedge.

---

## 2026-08-16 (cont.) — Kept convex hull as a stylistic option

User liked the visual look of the convex hull version despite it not being scientifically rigorous (see previous entry). Rather than lose it, kept both: `fetch_hull_meshes.py` now builds and saves both `smooth` (marching cubes) and `convex` (subdivided/Laplacian-smoothed convex hull) surfaces into the same `hull_meshes.json`, keyed separately. Both benefit from the `NO_CONS` exclusion fix, so the convex version should look better-proportioned than the original blade-shaped screenshot, not identical to it.

Added a `shell_style` parameter through `build_scene_html()` and a radio toggle in the app ("Smooth (density isosurface -- more anatomically meaningful)" vs "Convex hull (faceted, stylistic)"), so both are one click away without needing to touch code.

**Next:** confirm both styles render correctly live and pick whichever becomes the default going forward.

---

## 2026-08-16 (cont.) — Wired in mid/treble bands and a tonotopic-style split

User asked directly: "does this brain respond to the actual music track or does it just light up when music plays?" Honest audit of `simulate.py` found a real gap: the drive formula only used `rms`, `onset`, and `bass` -- `mid` and `treble` were extracted by `audio_features.py` but never fed into the simulation, and `tempo` was computed but only ever displayed as text.

**Important caveat stated to the user and in code comments:** FlyWire's public export has no per-neuron frequency-tuning annotation, so a *real* tonotopic map isn't available from the data. What was built instead is an explicitly-labeled modeling choice: a deterministic 3-way split of the ~4,502 seed neurons by `root_id % 3` (stable across runs), where each group is driven mainly by its assigned band (bass/mid/treble) plus a shared rhythm/loudness term everyone feels:
```
drive[band] = drive_scale * (0.3*rms + 0.2*onset + 0.5*band_energy)
```

**Verified with a synthetic test**, not just eyeballed: fed a signal that's pure bass for the first half and pure treble for the second half. Bass-group activation: 0.86 -> 0.35. Treble-group: 0.30 -> 0.86. Mid-group (no mid content in the test signal) stayed flat (~0.32-0.35) throughout, as expected. Confirms the groups genuinely differentiate by spectral content, not just react to "sound present."

Added a "Response by frequency band" chart in `app.py` so this is visible live for any real track, not just asserted -- if the three lines diverge over the course of a song, that's the spectral responsiveness actually working.

**Next:** try a real track and see whether bass/mid/treble activation visibly diverges over the course of the song (bassy sections vs. bright/high-passage sections should now produce visibly different group activation).

---

## 2026-08-16 (cont.) — Wired in real neurotransmitter signs + contained the camera

**Neurotransmitter signs (the #2 improvement from the earlier "how could we improve this model" discussion):** the connections file's `nt_type` column turned out to be entirely empty (all NaN) -- but `neurons.csv`'s "Predicted NT type" is a real per-neuron field, which is actually the biologically correct level anyway (transmitter identity is a property of the presynaptic neuron, not the individual synapse).

Sign convention taken from published literature, documented in `simulate.py`'s module docstring:
- ACH (acetylcholine) -> excitatory (nicotinic receptors, cation channels)
- GABA -> inhibitory (well established)
- GLUT (glutamate) -> inhibitory -- the non-obvious, citable one: unlike vertebrate CNS, most Drosophila glutamatergic synapses act through glutamate-gated chloride channels. Convention used in Lappalainen et al. 2024, *Nature* ("Connectome-constrained networks predict neural activity across the fly visual system").
- HIST (histamine) -> inhibitory (Hardie, 1989, *Nature* -- histamine-gated chloride channels at the photoreceptor synapse)
- DA/SER/OCT/TYR (biogenic amines) and unresolved predictions -> small neutral magnitude (0.3), since these have both excitatory and inhibitory receptor subtypes depending on target with no single universal sign in the literature -- explicitly not claiming false precision here.

Implementation: each edge weight is now signed by its presynaptic neuron's NT sign before the sparse matrix is built. Normalization switched from raw sum to sum-of-absolute-value in-strength, so excitatory/inhibitory inputs can genuinely cancel in the actual signal rather than in the normalizer.

**Verified, not just asserted:** 24.2% of edges are now inhibitory (matches the real GABA+glutamate+histamine proportion of predicted types in our subgraph). A test run showed 32% of neurons with genuinely negative (suppressed) final activation, range -0.85 to +0.95 -- real inhibition dynamics, not just varying brightness on an all-positive scale.

**Camera containment:** user reported the scene starts too zoomed out and the orbit space feels too vast. Tightened initial distance (1.5x -> 0.85x structure radius) and added hard `controls.minDistance`/`maxDistance` (0.35x-1.8x radius) so users can no longer scroll out into empty space.

**Next:** confirm live that the camera feels contained now, and that inhibition is visually apparent (some regions/neurons going dark/suppressed rather than everything only ever getting brighter).

---

## 2026-08-16 (cont.) — Fixed off-center rendering, lit-before-playing, and locked camera

**Disabled auto-rotate** (`controls.autoRotate = false`) per request -- the model now stays fixed in place until manually dragged, instead of continuously spinning.

**Off-center bug:** user asked why the model renders on the right side instead of centered. Real cause: the scene measured `container.clientWidth` once at script start to set the camera's aspect ratio, but Streamlit's iframe often hasn't finished laying out to its actual full width at that point -- so the aspect ratio got locked in wrong, and the only recovery path was a `window resize` event, which a user dragging/orbiting the model would never trigger. Replaced with a `ResizeObserver` on the container that recalculates aspect/renderer size whenever the real size settles, not just on window resize.

**Lit-up-before-playing bug:** user asked why neurons were already glowing before pressing Play. Real cause: the idle/pre-play state was displaying `DATA.frames[0]` -- frame 0 of the fully precomputed simulation, which already reflects real activation from the very start of the track (the whole thing is computed offline ahead of time, not live). Fixed by forcing the true idle state to the dark end of the color scale (`DATA.cmin`) instead of showing precomputed frame 0, and changed the pause/stop fade to decay toward `cmin` rather than toward a raw zero that could land mid-scale if `cmin` is negative (which it now regularly is, since neurotransmitter signs were wired in).

**Next:** confirm live that the model starts centered, starts dark/off before playing, and stays fixed in place until dragged.

---

## 2026-08-16 (cont.) — Fixed real centering bug (was targeting bounding-box midpoint, not center of mass)

Previous "off-center" fix (ResizeObserver) didn't actually solve it -- user screenshot confirmed the object still rendered in the upper-right, well off the true center of the frame, even after that fix. Correctly diagnosed the real cause this time: `computeBounds()` set the camera target to the midpoint of the combined bounding box (min+max)/2 -- but the nerve cord hull extends much further in one direction than the brain hull or the actual glowing neuron cluster, so the box midpoint sits in relatively empty space, away from where the dense, visible structure actually is. Camera math itself was fine; it was correctly centering on an *empty* point.

Fixed by targeting the actual centroid of `fgPositions` (the real auditory-pathway neurons -- the actual subject of the visualization) instead of the bounding-box midpoint. Bounding radius (for camera distance/zoom limits) still comes from the full combined extent, so the whole structure stays in frame.

**Lesson:** bounding-box center and center-of-mass are not the same thing, and conflating them is a classic 3D scene-framing bug -- worth remembering for any future viewport-fitting code.

---

## 2026-08-16 (cont.) — Fixed instant-snap activation, nudged framing down, restored orbit

**Instant light-up bug:** user reported activation jumped instantly on pressing Play instead of fading in with the music. Real cause: the animate loop was directly *assigning* `currentDisplay = interpolatedValues(currentTime)` each frame while playing -- no smoothing, so it snapped straight to whatever the target value was. Fixed by easing toward the target (`currentDisplay[i] += (target[i] - currentDisplay[i]) * 0.12` per frame) instead of overwriting, so activation now genuinely ramps up/down relative to the track rather than snapping.

**Framing:** user confirmed the shell-centroid fix (previous entry) got horizontal/depth positioning right, and asked to nudge it down the Y axis. Implemented by looking slightly above the model's true center (`lookY = center.y + radius*0.18`) -- since OrbitControls always renders its target dead-center, aiming above the model pushes the model itself lower in frame.

**Orbit restored:** re-enabled `controls.autoRotate` (slow, 0.5 speed) now that framing is correct -- was disabled earlier at user request when the model was still mispositioned and constant motion made it hard to evaluate.

**Next:** confirm live that activation fades in smoothly with the track, the model sits lower in frame as intended, and the slow rotation reads well now that framing is fixed.

---

## 2026-08-16 (cont.) — Real leaky integrate-and-fire dynamics (the #1 improvement)

Checked user-provided FAFB downloads (`classification.csv.gz`, `fafb_v783_princeton_synapse_table.csv.gz`, `neurons.csv.gz`) and the BANC<->FAFB cross-registration table (`banc_fafb_nblast_v2`) as a possible bridge to richer per-synapse NT data. Decided against using it: match confidence averages ~0.86 (NBLAST morphological similarity, not synaptic identity) and 90% of matches are unvalidated -- chaining two ~85%-confidence cross-dataset guesses (source match + target match) to get per-synapse precision isn't a good trade against what we already have (real per-neuron predicted NT, ~93% coverage, already wired in). Also confirmed the FAFB Neuron Skeletons file (13GB) is unsuitable regardless -- both because of size (our current architecture inlines everything as one HTML blob, already ~15MB) and the same cross-dataset ID mismatch.

**Implemented real LIF dynamics** -- the #1 item from the earlier "how to improve this model" priority list. Replaced the generic `tanh(decay*state + drive + alpha*incoming)` recurrence with actual leaky integrate-and-fire neurons, Euler-integrated:
```
tau_m * dV/dt = -(V - V_rest) + I_syn(t) + I_drive(t)
```
with real spiking (threshold crossing -> reset -> refractory period), synaptic current driven by actual presynaptic spikes (not a continuous "activation" abstraction), and the existing NT-sign matrix providing genuine excitatory/inhibitory currents.

`tau_m = 20ms` is cited from Gouwens & Wilson 2009 (*J Neurosci*, "Signal propagation in Drosophila central neurons"), who directly measured fly central neuron membrane time constants (~10-20+ ms) -- a real, sourced parameter, not an arbitrary guess. Documented clearly in the module docstring: this is a representative value, not a per-neuron fitted one (FlyWire's data has no per-neuron electrophysiology), and the audio-frame-to-substep resolution trade-off (25 Euler substeps per audio frame, ~1s total runtime, vs. true continuous ms-resolution integration across a multi-minute track, which isn't tractable for an interactive local app) is stated plainly rather than implied to be exact.

**Verified, not just implemented:** re-ran the bass/treble-half synthetic test -- differentiation is now sharper (0.75 vs ~0.0006, essentially binary) and shows a genuine LIF signature (rapid convergence to a steady firing plateau under constant input, characteristic of real spiking dynamics, not a smooth tanh ramp). Full real-track run: ~1 second, no NaNs, membrane potential floor at exactly 0.0 for heavily inhibited/quiescent neurons (emergent, not clamped in code).

**Next:** confirm live that the model still feels responsive/visually compelling with the new dynamics, and decide whether any further items from the priority list (dropping the fake tonotopic split, or explicitly labeling it as artistic license) are worth doing next.

---

## 2026-08-16 (cont.) — Fixed "lit before playing" and "instant" activation

User: "why do the neurons fire up before the music plays, and instantly?" Two separate real bugs, not one:

1. **Color scale anchoring bug (the "lit before/instantly" root cause):** `cmin` (the dark end of the color scale) was set to the 2nd percentile of all observed activation values across the whole run. Since inhibited neurons can go negative, this pulled `cmin` below zero -- but most neurons sit at exactly their resting potential (V=0, quiescent) most of the time. With `cmin < 0`, that resting value of 0 was already mapping to a visibly non-zero, lit color. The entire population looked "on" the moment any dynamics started, even neurons that were doing nothing. Fixed by anchoring `cmin` to true rest (0.0) rather than a data-derived percentile -- quiescent now genuinely reads as dark.

2. **Snapshot resolution too coarse:** `node_snapshot_stride=10` meant a visual update only every 10 audio-frames -- for a real ~4:30 track (150 total frames spanning the whole song), that's a snapshot roughly every **18 seconds**. Between two widely-spaced steady-state samples 18s apart, any transition necessarily looks like a jump rather than a gradual response to the music. Reduced stride to 3 (~5.4s between snapshots, 51 total instead of 15), payload grew modestly (10.6MB -> 16.7MB) but sim time stayed ~1.1s.

**Next:** confirm live that quiescent neurons now stay dark before/between activity, and that the ramp-up on play now visibly tracks the track's actual frequency/rhythm changes rather than jumping.

---

## 2026-08-16 (cont.) — Found and fixed real bug: works on noise, dead on real songs

User: "works on white/pink noise but not when i drag and drop a song." Verified directly with browser automation (Playwright) that the pipeline genuinely works correctly on the pink noise control -- idle dark, clear bright activation cluster mid-playback, no console errors. So the bug was track-specific, not a rendering/pipeline failure.

Isolated methodically: MP3 decoding path tested in isolation with a converted pink-noise-as-MP3 file -- worked fine, ruling out format/ffmpeg issues. That left track *length* as the remaining variable (our test files are 30s; a real song is 3-5 minutes).

**Root cause, reproduced directly:** the LIF simulation used a FIXED substep count (25) per audio frame, but the actual Euler integration timestep is `frame_duration / substep_count`. For a 30s test clip (150 frames total), that's ~8ms -- safely below the 20ms membrane time constant. For a real ~4.5 minute song, each audio frame stretches to ~1.8-2s, pushing the timestep to ~70-80ms -- *larger* than tau_m. Euler integration is only numerically stable while dt stays well below the time constant; once dt exceeds it, each step overshoots rather than converging, causing oscillation. That oscillation averages out to near-zero (and sometimes flips negative) in the per-frame reported value -- exactly matching what looked like "nothing lights up." Reproduced synthetically: the same drive pattern that gave 0.88 max activation at 30s duration gave only 0.32 max (and negative seed trace) at a 270s duration, before the fix.

**Fix:** derive substep count from a fixed target dt (5ms, safely under tau_m) instead of a fixed count -- `n_substeps = frame_duration_ms / TARGET_DT_MS`, capped at 800/frame to bound worst-case runtime on unusually long tracks. Verified: the 270s synthetic case now gives 0.82 max activation (back to healthy range, matching the 30s case), runtime ~10s for a 4.5-minute track (acceptable with the existing spinner), and short clips are unaffected (~1.5s).

**Lesson:** any fixed-iteration-count numerical integration scheme needs to be checked against its actual timestep size across the full range of expected inputs, not just the size used during initial testing -- this bug was invisible in every test so far because all our test audio was short.

---

## 2026-08-16 (cont.) — Adopted Shiu et al. 2024's exact published LIF parameters

User shared the Nature "FlyWire connectome" collection page. Most relevant entry: Shiu et al. 2024, Nature, "A Drosophila computational brain model reveals sensorimotor processing" -- a leaky integrate-and-fire model built on this SAME FlyWire connectome. Directly answers the earlier-flagged "arbitrary" parts of our model with real published numbers.

Adopted their exact values (previous ad hoc/unitless choices in parens):
- tau_m = 20ms = R_mbr(10 MOhm) x C_mbr(0.002 uF) -- independently matches our own earlier Gouwens & Wilson 2009 citation almost exactly, a genuine cross-validation between two different sources.
- V_rest = -52mV, V_th = -45mV (7mV gap), V_reset = -52mV (previously unitless 0/1/0)
- T_refrac = 2.2ms (previously 2.0ms, close already)
- **Synapse-to-weight conversion switched from log1p-normalized/unitless to LINEAR**: `weight = raw_synapse_count * sign * W_syn`, with `W_syn = 0.275mV` as their fitted single free parameter. This was the most-flagged "not from literature" gap in the earlier improvement list -- now directly traceable to a peer-reviewed source.
- Removed the "normalize incoming by absolute in-strength" step -- that was our own workaround for the old unitless model's saturation problem; Shiu et al.'s real methodology sums raw signed weighted contributions directly, no such normalization.
- `alpha` (synaptic current scaling) default changed from 2.2 to 1.0, matching their formula exactly with no extra scaling.
- `I_drive` (the audio-to-current mapping) has no literature source and remains our own artistic layer -- rescaled to a sensible mV magnitude (default 12mV, sized against the real 7mV threshold gap) and explicitly distinguished in the docstring from the now-literature-sourced synaptic/membrane parameters.
- `state` is reported RELATIVE TO REST (not raw mV) so all downstream consumers (coloring, charts, "activation" values) keep their existing "0 = rest, + = depolarized" semantics unchanged.

**Verified stable, not just implemented:** real synapse counts in our subgraph range up to 913 on a single edge (99th percentile 70) -- checked this wouldn't cause runaway saturation with the new linear (not log-compressed) weighting. Result: healthy spread (41% of values show real activity, only 0.6% near max -- not everything saturating), median exactly at rest as expected. Re-verified the earlier long-track timestep-stability fix still holds under the new parameterization (270s synthetic track: max ~6.9mV relative to rest, no washout, ~9.6s runtime).

**Next:** confirm live that the model still feels responsive and visually compelling with the real published parameters, on both short test tracks and a real song.

---

## 2026-08-16 (cont.) — Extended subgraph to 3 hops (real reach, not a visual trick)

User asked whether the "only the bottom lights up" result was more scientifically accurate than extending further. Clarified: neither is more/less accurate -- both use real wiring data. The 2-hop cutoff was a computational choice (subgraph extraction limit), not a biological boundary; real auditory-evoked signal in an actual fly brain does propagate further than 2 synapses. Extending hops is equally real, just more complete. User chose to extend.

Tested 3-hop expansion at the original MIN_SYN=5 threshold first: 69,051 neurons, 552k edges -- too large for the current architecture (inline JSON payload in one HTML blob). Tested several MIN_SYN thresholds at 3 hops to find a practical size; landed on **MIN_SYN=12** (up from 5): 22,878 neurons, 75,947 edges -- close to the previous 2-hop size, but now reaching a real hop further and keeping only stronger (>=12 synapse) connections, which is arguably more defensible too (drops noisy weak connections).

Rebuilt the full downstream pipeline: `fetch_connectome.py` (new subgraph) and `fetch_coordinates.py` (real 3D positions for the new neuron set, 99.1% coverage maintained). Background context sample and brain/cord hull shells are independent of the subgraph (whole-brain random sample), so those didn't need rebuilding.

**Verified:** subgraph now spans 51 real anatomical regions (up from 41), including meaningfully more MB_CA (mushroom body calyx) and ME (medulla) representation -- the regions that were previously only faintly present. Simulation still fast (~1s), stable, healthy activation spread (34% >0.5, no runaway saturation). Scene payload actually shrank slightly (23MB -> 19MB) since the higher synapse threshold offset the extra hop's neuron count.

**Next:** confirm live that more of the brain (not just the bottom cluster near AMMC/GNG) now shows visible activity.

---

## 2026-08-16 (cont.) — Fixed: clicking shell-style toggle wiped the whole results view

User: "this option does not work" (Convex hull shell-style radio). Reproduced directly with Playwright: clicking it reset the entire app back to the blank pre-run form, losing all charts and the scene.

Root cause: classic Streamlit pitfall. `st.button("Run simulation")` only returns `True` on the exact script execution where it was clicked -- on every subsequent rerun (triggered by touching ANY other widget, including the shell-style radio placed later in the script) it returns `False` again. The entire results section lived inside `if run:`, so touching any widget after the initial run caused that whole block to evaluate False and vanish.

Fixed by separating computation from rendering: the simulation still only runs `if run:` (on button click), but the result, features, audio bytes, and source name are now stashed in `st.session_state`. Rendering happens `if "result" in st.session_state:`, independent of the button's momentary state -- so later widget interactions (shell style, view toggle) just re-render from the persisted result instead of wiping it.

**Verified via Playwright, not just inferred:** ran the pink noise control, clicked "Convex hull," confirmed the full results view (charts, scene, everything) survived and the radio's selection stuck. Bonus observation from the same screenshot: the frequency-band chart now shows clear, sustained divergence between bass/mid/treble groups (bass consistently ~4-7, mid ~2-3, treble ~1-2) -- the tonotopic-style split is visibly working well under the new Shiu et al. parameters.

---

## 2026-08-16 (cont.) — Boosted visual brightness mapping + explained angle-dependent glow

**Visual brightness boost (rendering-only, no change to underlying simulation):** user felt neurons weren't "firing up" visually as much after the Shiu et al. parameter switch. Real LIF dynamics are genuinely sparser/more graded than the old continuous tanh model, and Inferno reads dark below ~50% -- so moderate (but real) activity was landing in visually dim territory. Added a `sqrt` curve to the brightness/size mapping in `web_scene.py` (lifts midtones without changing relative ordering or the underlying values) and tightened the color scale reference point (95th percentile instead of 98th). Purely a perceptual remapping, not a data change.

**Angle-dependent brightness, explained not fixed:** user noticed the glow looks brighter/more saturated from some camera angles than others. This is an expected property of additive-blended point rendering, not a bug: the underlying activation values don't change with camera angle, but additive blending sums the color of every point that overlaps in screen space. Viewed edge-on (along the cluster's long axis), many points project onto nearly the same pixels and their brightness accumulates into a hot core; viewed from another angle, the same points spread across more pixels and each gets less accumulated light. This is inherent to the "hot glow" look (it's part of why additive blending was chosen), not something wrong with the simulation. Presented the tradeoff (leave it / reduce opacity for more consistency but less punch / switch off additive blending entirely and lose the glow look) -- user chose to leave it as-is.

---

## 2026-08-16 (cont.) — UX/UI cleanup pass

User asked for UX/UI recommendations, then approved implementing them. Changes:

1. **Dark theme** (`.streamlit/config.toml`, `base="dark"` + custom palette matching the scene's own colors) -- previously the app ran in Streamlit's default light theme while the centerpiece is a black glow scene, so it looked like a dark video embedded in a mismatched light dashboard. Now one cohesive dark page.
2. **Reordered so the scene surfaces immediately** after running -- previously you scrolled past two diagnostic charts (activation-over-time, frequency-band response) before reaching the actual visual. Those charts plus the "most activated neurons" table are now tucked into a collapsed `st.expander("Simulation details...")` below the scene, available on demand but not blocking the main view.
3. **Full-width layout** -- removed the `col1`/`col2` split that left an empty right column doing nothing before results existed; uploader and control-track selector now span full width with the run button alongside.
4. **Trimmed redundant captions** -- the in-scene legend already explains the color/shape meaning and interaction hints, so removed the duplicate outer Streamlit captions that repeated the same information.

Verified visually via Playwright screenshots (before-run and after-run states) -- clean full-width dark layout, scene appears right after the view/shell toggles with no diagnostic clutter in between.

---

## 2026-08-16 (cont.) — Verified-NT preference + About panel

Both remaining items from the earlier "next steps" discussion:

**1. Prefer verified over predicted NT type.** `neurons.csv` has both "Predicted NT type" (broad coverage) and "Verified NT type" (higher confidence, smaller coverage, sometimes multi-transmitter e.g. "gaba,nitric_oxide" for co-release). `_load_nt_types()` now resolves per-neuron: verified first (taking the first-listed transmitter for multi-transmitter entries, documented as a simplification -- real co-release can't be represented by a single sign), falling back to predicted otherwise. Verified, not just implemented: re-ran the full pipeline, 151,823 neurons resolved, healthy activation stats unchanged (min -43.7, max 6.9, 34% >0.5) -- confirms nothing broke.

**2. "About this project" panel.** Added a collapsed `st.expander` right below the title, laying out plainly what's real (BANC v888 connectome, real NT-based excitatory/inhibitory signs with citations, real LIF dynamics from Shiu et al. 2024) vs. speculative (the audio-to-drive mapping, the tonotopic band split) -- previously this rigor was scattered across code docstrings and small captions, invisible to anyone viewing the app without reading the source. Verified rendering via Playwright screenshot -- clean, readable, expands correctly.

Both changes visible live at localhost:8501.

---

## 2026-08-16 (cont.) — Button alignment fix

User flagged the "Run simulation" button wasn't vertically aligned with the "...or use a control track" selectbox next to it -- the `st.write("")` spacer used to push it down wasn't the right height. Replaced with a precise `st.markdown("<div style='height:28px'></div>")` spacer. Verified via Playwright screenshot: button now lines up exactly with the selectbox input.

---

## 2026-08-19 — Audio front-end overhaul: mel bands, onset, HPSS, chroma, beat-sync

Prompted by a comment on the LinkedIn post ("cochlear-style front end with mel spectrograms and log compression might help the network respond to real music dynamics") -- used as the jumping-off point for a broader pass on the audio-to-drive mapping layer, planned and shipped as a sequence of small, independently revertable commits.

**1. Mel spectrogram + log compression, 8 bands (was: 3-way linear bass/mid/treble split).** `audio_features.py` now uses `librosa.feature.melspectrogram` (denser bins at low frequencies, matching real cochlear/Johnston's Organ tuning) with `power_to_db` log compression, collapsed into `N_BANDS=8` contiguous bands instead of a flat 3-way linear-Hz split. `simulate.py`'s `band_group` generalized from a hardcoded `% 3` to `% N_BANDS`.

**2. Per-band onset detection.** Previously one whole-mix onset envelope was shared by every band, so a kick drum and a cymbal hit rode the same global transient bump. `onset_strength` now runs per mel-band slice (reusing the already-log-compressed `mel_db`), so each band gets its own local transient signal.

**3. HPSS (percussive/harmonic separation).** `librosa.effects.hpss` splits the mix before feature extraction. The per-frame percussive-vs-harmonic ratio now modulates drive blending in `simulate.py`: percussive-dominated moments lean harder on the sharp per-band onset term, harmonic-dominated moments lean harder on the smoother band-energy term -- two structurally different response characters instead of one blended signal for both.

**4. Chroma-driven pitch-class population.** A second, independent 12-way seed grouping (`pitch_group = node % 12`) driven by chroma (which of the 12 pitch classes are sounding, computed on the HPSS harmonic component). First implementation stacked this as a second additive drive term on top of the band term -- user feedback: "the added seed grouping makes the model brighter, kinda loses clarity." Root cause: two independent full-strength drive signals summed on the same neurons just raises the floor everywhere instead of picking out which neurons respond harder. Fixed by making chroma a multiplier (`[0.7, 1.3]` factor) on the existing band drive instead of a second additive term -- confirmed "bit better" after the fix.

**5. Brightness/rendering fixes (two rounds), found from screenshots, not guessed.** User: "still gets bright at the bottom" (screenshot showed two saturated white blobs near Johnston's Organ/AMMC). Diagnosed empirically, not assumed: pulled the actual seed-layer trace and found 77% of frames sat above 50% of the layer's own peak -- a real structural skew (seed neurons receive raw drive directly, no synaptic filtering, so they're elevated most of the track), compounded by the existing `sqrt()` render curve that *lifts* midtones further toward white, the opposite of what was needed. Round 1: switched color/size normalization from one global scale to per-hop-group (seed/hop1/hop2+) scales, each with its own gamma curve (seed compressed at exponent 1.6, downstream groups keep the original 0.5 boost). User: "bit better but still bright at the bottom" -- a second screenshot showed two localized clusters still pinned white, distinct from the general seed-layer issue. Diagnosed as individual "hub" neurons (heavily-connected downstream targets) sitting near their own ceiling almost constantly, which a group-level scale can't catch since they're part of what defines that group's ceiling. Round 2: switched from per-group to per-node normalization -- every neuron's own 95th-percentile range across the track sets its own ceiling. User confirmed "looks better" after this round.

**6. Opt-in beat-synced timing.** Locks simulation frames to the track's detected beat grid (subdivided, off by default via checkbox) instead of a fixed wall-clock rate. Required moving `simulate.py`'s substep/dt computation from once-globally to inside the per-frame loop, since beat-synced frames have genuinely irregular duration (tempo drift, rubato) and a global dt estimate would silently destabilize on any frame longer than that estimate -- the same numerical-instability class already documented in this file's earlier LIF-tuning entries. Self-audited afterward (asked to "check for any bugs or tweaks that need adjusting" before considering it done) and found one real issue: an uncapped beat grid on a long, brisk-tempo track (the full 4:31 Chopin demo) produced 887 frames vs. fixed mode's 150, bloating the embedded HTML payload and node-snapshot count proportionally. Added `MAX_BEAT_FRAMES=300` with even thinning, mirroring the existing `MAX_SUBSTEPS_PER_FRAME` guard philosophy. Also found the UI's checkbox help text overpromised: it claimed the Chopin demo would "fall back to fixed-rate timing" as a rubato example, but empirical testing showed librosa's beat tracker detects a usable pulse on it (67 beats), so beat-sync engages there too -- copy corrected to describe the fallback accurately (only triggers on very-few-beats material like short clips or silence).

Every step in this pass was implemented, pipeline-tested (extract → simulate, checked for NaN/instability), and boot-tested in a live Streamlit run before committing -- each shipped as its own small, separately revertable commit rather than one large batch.
