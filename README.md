# Fly Brain on Music

Drop in a track and watch it propagate through a **real fruit fly connectome**.

This app maps audio onto the real FlyWire/CAVE **BANC v888** connectome (~23k neurons, ~76k real synaptic connections downstream of the fly's actual auditory sensory organ) and simulates activation with a real **leaky integrate-and-fire (LIF)** spiking model, using published parameters from Shiu et al. 2024, *Nature*. The result is rendered as an interactive, audio-synced 3D scene.

The wiring, the neurotransmitter signs, and the spiking dynamics are real, cited science. The audio-to-neural-drive mapping is our own artistic modeling layer: no dataset of real fly neural response to music exists to validate against. Treat this as **connectome-constrained generative art**, not a research claim about how flies perceive music. The app's own "About this project" panel spells out exactly what's real vs. speculative, line by line.

<p align="center">
  <img src="screenshots/scene-overview.png" width="90%" alt="Full connectome scene, glowing auditory pathway activation">
</p>

<table align="center">
  <tr>
    <td><img src="screenshots/scene-closeup-1.png" width="100%" alt="Close-up of activated neuron clusters, brain shell"></td>
    <td><img src="screenshots/scene-closeup-2.png" width="100%" alt="Close-up of activated neuron clusters, alternate angle"></td>
  </tr>
</table>

## What's real vs. speculative

**Real, from published data:**
- **Connectome**: FlyWire/CAVE's BANC v888 dataset, real neurons, real synapses, real 3D positions. The subgraph is every neuron within 3 hops downstream of Johnston's Organ (the fly's real auditory sensory organ) → AMMC, filtered to connections with ≥12 synapses.
- **Neurotransmitter identity & sign**: from each neuron's verified (preferred) or predicted NT type. Acetylcholine is excitatory; GABA, glutamate, and histamine are inhibitory, following the fly-specific convention (unlike vertebrate CNS) used in Lappalainen et al. 2024, *Nature*, and Hardie, 1989, *Nature*.
- **Neuron dynamics**: a real LIF model with published parameters from Shiu et al. 2024, *Nature*, a computational brain model built on this same connectome. Its membrane time constant (20ms) independently matches Gouwens & Wilson 2009's direct electrophysiological measurement.

**Speculative, our own modeling layer:**
- **Audio → neural drive**: loudness, onset, and frequency-band energy mapped to synaptic current. An artistic choice, not a validated model.
- **"Tonotopic" band split**: FlyWire's public data has no per-neuron frequency-tuning annotation, so the bass/mid/treble seed grouping is a deterministic but arbitrary partition, not a real tonotopic map.

## Try it

A public-domain Chopin recording is bundled as the default demo track, so just hit **Run simulation**: no upload needed. White/pink noise control tracks are also available for verifying the simulation reacts to input at all.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

The FlyWire connectome subgraph, neuron positions, and demo audio are already included in `data/`, so this works immediately. No CAVE account or auth token is needed just to run the app.

## Regenerating the data from scratch

Only needed if you want to rebuild the connectome subgraph yourself (different hop radius, synapse threshold, etc.) rather than use what's committed in `data/connectome/`.

```bash
pip install -r requirements-pipeline.txt
python src/get_token.py          # one-time: FlyWire/CAVE auth token
python src/check_auth.py         # verify it worked
python src/fetch_connectome.py   # pull the auditory-pathway subgraph
python src/fetch_coordinates.py  # real 3D neuron positions
python src/fetch_background_positions.py
python src/fetch_hull_meshes.py
```

Requires your own FlyWire/CAVE account. See [codex.flywire.ai](https://codex.flywire.ai).

## How it works

1. **`audio_features.py`**: extracts tempo, RMS loudness, onset strength, and bass/mid/treble band energy from the uploaded track (librosa).
2. **`simulate.py`**: drives the real connectome graph with those features. Each of the ~23k neurons is a leaky integrate-and-fire unit; real synaptic weights (linear in synapse count, signed by neurotransmitter) propagate spikes hop by hop from the seed (Johnston's Organ) neurons outward. See the module docstring for the full model writeup, including the resolution trade-offs made for interactivity.
3. **`web_scene.py`** + **`brain_map_3d.py`**: renders the result as a three.js scene using each neuron's real 3D position (from CAVE), colored and sized by simulated activation, audio-synced frame by frame.
4. **`brain_map.py`**: a standalone, non-interactive 2D schematic fallback (hand-placed neuropil coordinates, not real 3D positions). Not wired into the app, kept as a lightweight backup visualization.

## Build story

This started as a scaffolding-only idea (pull a connectome, drive it with audio, see what happens) and turned into a lot of dead ends and real bug hunts along the way. The highlights, roughly in order:

- **Data access.** The original plan was CAVEclient auth against FlyWire's production dataset. That path stalled, so the connectome ended up being pulled via manual download instead, and the first real graph got built from there.
- **Pivot to drag-and-drop.** Early versions expected pre-stored audio files per genre. That got scrapped in favor of a simple drag-and-drop uploader, no stored files, so anyone could try their own track.
- **Finding real 3D coordinates.** The first visualization was a hand-authored schematic (now `brain_map.py`), since no 3D mesh data seemed available from the public export. Real per-neuron 3D positions turned up later in CAVE's `cell_representative_point` table, which unblocked a true 3D visualization and made the schematic a fallback instead of the main view.
- **Real leaky integrate-and-fire dynamics, called the #1 improvement in the build log.** The activation model went through several iterations (a generic recurrence formula, then various tanh-squashed approximations) before landing on an actual LIF spiking model, then later adopting Shiu et al. 2024's exact published parameters for it, giving the simulation a real, cited membrane time constant instead of an arbitrary one.
- **Real bugs, not just tuning.** A few genuine correctness bugs got caught and fixed along the way: activation that snapped instantly instead of following the membrane dynamics, a centering bug from targeting the bounding-box midpoint instead of center of mass, a case where the simulation worked on synthetic noise but stayed dead on real songs, and a geometry bug behind an oddly blade-shaped brain hull.
- **Extending reach.** The auditory subgraph started at 2 hops downstream of Johnston's Organ. It later got extended to a real 3 hops, tuning the synapse-count threshold up to keep the neuron count computationally tractable while reaching further into the real wiring.
- **UX passes.** Several rounds of cleanup followed once the core simulation was solid: a Streamlit `session_state` bug where touching any widget after running the simulation wiped the whole results view, a dark theme pass to match the scene instead of clashing with Streamlit's default light UI, a loading indicator that nods to the real FlyWire/CAVE segmentation viewer's look, and finally moving the transport controls to sit outside the 3D viewport instead of overlapping it.

The full, unedited development log lives in [`BUILD_LOG.md`](BUILD_LOG.md) if you want the blow-by-blow.

## Credits

- Connectome data: [FlyWire](https://flywire.ai) / [CAVE](https://www.cave-connectome.org), BANC v888 dataset.
- LIF model parameters: Shiu et al. 2024, *Nature*, "A *Drosophila* computational brain model reveals sensorimotor processing."
- Neurotransmitter sign convention: Lappalainen et al. 2024, *Nature*; Hardie, 1989, *Nature*.
- Demo track: Chopin, *Nocturne in E-flat major, Op. 9 No. 2*, performed by Frank Levy, public domain (CC0) via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Nocturneop9no2-.ogg).

## License

Code: MIT. See [LICENSE](LICENSE). The connectome data in `data/connectome/` is derived from FlyWire/CAVE and subject to [FlyWire's own data usage terms](https://codex.flywire.ai/api/download), not this repo's MIT license.

## Disclaimer

This is a speculative/artistic simulation, not a validated behavior predictor. The connectome and spiking model are real and cited; the audio-to-neuron-drive mapping and the resulting "behavior" are a modeling choice, not established science.
