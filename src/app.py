"""Drag-and-drop interactive app: audio in, real FlyWire connectome reacts.

Run with: streamlit run src/app.py
Nothing uploaded here is saved to disk or sent anywhere else -- audio is
read into memory, processed, and discarded when the session ends.
"""
import io
import mimetypes
from pathlib import Path

import librosa
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from audio_features import extract_features
from em_texture import EM_LOADER_VIDEO_B64
from simulate import run_simulation
from web_scene import build_scene_html

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "audio"

st.set_page_config(page_title="Fly Brain on Music", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 3rem !important; }
      /* brighten low-contrast helper text (captions, help tooltips, uploader hints) */
      [data-testid="stCaptionContainer"] p,
      [data-testid="stFileUploaderDropzoneInstructions"] div,
      [data-testid="stWidgetLabel"] p {
        color: #b6b6c2 !important;
      }
      /* accent the About expander to match the purple/gold theme instead of default gray */
      [data-testid="stExpander"] { border-color: #3a3a4a !important; }
      [data-testid="stExpander"] summary { color: #d8d8e0 !important; }
      [data-testid="stExpander"] summary svg { color: #8f5ad6 !important; }
      /* make the selected file read as "this is what will run", not inert text */
      [data-testid="stFileUploaderFile"] {
        border-left: 3px solid #8f5ad6;
        background: rgba(143,90,214,0.08);
        border-radius: 4px;
        padding: 4px 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

ABOUT_MD = """
**Real, from published data:**
- **Connectome**: FlyWire/CAVE's BANC v888 dataset — real neurons, real synapses, real 3D positions.
  Subgraph = every neuron within 3 hops downstream of Johnston's Organ (the fly's real auditory
  sensory organ) → AMMC, filtered to connections with ≥12 synapses.
- **Neurotransmitter identity & sign**: from `neurons.csv`'s verified (preferred) or predicted NT type
  per neuron. Acetylcholine = excitatory; GABA, glutamate, and histamine = inhibitory — the glutamate/
  histamine calls are the non-obvious ones: unlike vertebrate CNS, *Drosophila* glutamate mostly acts
  through inhibitory glutamate-gated chloride channels (convention per Lappalainen et al. 2024, *Nature*),
  and histamine is inhibitory via histamine-gated chloride channels (Hardie, 1989, *Nature*).
- **Neuron dynamics**: real leaky integrate-and-fire equations with published parameters from
  Shiu et al. 2024, *Nature* — a computational brain model built on this same connectome. Membrane
  time constant (20ms) independently matches Gouwens & Wilson 2009's direct electrophysiological
  measurement, a genuine cross-validation between two different sources.

**Speculative, our own modeling layer — not from literature:**
- **Audio → neural drive**: no dataset of real fly neural response to music exists anywhere, so this
  mapping (loudness/onset/frequency-band energy → synaptic current) is an artistic choice, not a
  validated model.
- **"Tonotopic" band split**: FlyWire's public data has no per-neuron frequency-tuning annotation, so
  the bass/mid/treble neuron grouping is a deterministic but arbitrary partition (by neuron ID), not a
  real tonotopic map.

**Bottom line**: this is a real wiring diagram with a real spiking-neuron model running on top,
driven by an artistic (not scientifically validated) audio-input layer. Treat it as connectome-
constrained generative art, not a research claim about how flies perceive music.
"""

CONTROL_TRACKS = {
    "Chopin – Nocturne Op. 9 No. 2 (demo)": "chopin_nocturne_op9_no2.mp3",
    "white noise": "noise_white.wav",
    "pink noise": "noise_pink.wav",
}

# A real (screen-recorded) clip of the FlyWire/CAVE segmentation viewer, looped
# inside a small status card -- shown during both processing stages in place of
# a generic spinner. `{stage_label}` is filled in per-stage by render_loader().
_EM_LOADER_TEMPLATE = """
<style>
  @keyframes proc-pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.35; transform: scale(0.65); } }
  .proc-card-overlay {
    position: fixed; inset: 0; z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    pointer-events: none;
  }
  .proc-card {
    display: inline-flex; flex-direction: column; gap: 10px;
    padding: 14px 16px;
    background: #0a0a12; border: 1px solid #232330; border-radius: 10px;
    box-shadow: 0 0 0 1px rgba(143,90,214,0.08), 0 6px 24px rgba(0,0,0,0.45);
    font-family: sans-serif;
    pointer-events: auto;
  }
  .proc-card__status { display: flex; align-items: center; gap: 8px; }
  .proc-card__dot {
    width: 8px; height: 8px; border-radius: 50%; background: #8f5ad6; flex: none;
    box-shadow: 0 0 8px 2px rgba(143,90,214,0.7);
    animation: proc-pulse 1.3s ease-in-out infinite;
  }
  .proc-card__label { color: #d8d8e0; font-size: 13px; font-weight: 600; letter-spacing: 0.2px; }
  .proc-card__video {
    position: relative; width: 200px; height: 116px;
    border-radius: 8px; overflow: hidden; border: 1px solid #2c2c38; background: #b8b8bc;
  }
  .proc-card__video video {
    width: 100%; height: 100%; object-fit: cover; display: block;
    filter: contrast(1.05) brightness(0.96);
  }
  .proc-card__vignette {
    position: absolute; inset: 0; pointer-events: none;
    box-shadow: inset 0 0 0 1px rgba(143,90,214,0.15), inset 0 0 26px rgba(0,0,0,0.4);
  }
  .proc-card__live {
    position: absolute; top: 6px; right: 7px; display: flex; align-items: center; gap: 4px;
    padding: 2px 6px; border-radius: 3px; background: rgba(10,10,18,0.55);
    font-size: 9px; letter-spacing: 0.5px; color: #ddd;
  }
  .proc-card__live-dot { width: 5px; height: 5px; border-radius: 50%; background: #e0554f; animation: proc-pulse 1.3s ease-in-out infinite; }
  .proc-card__caption {
    color: #6e6e7a; font-size: 10.5px; letter-spacing: 0.3px; text-transform: uppercase;
  }
  .proc-card__hint {
    color: #55555f; font-size: 11px; font-style: italic;
  }
</style>
<div class="proc-card-overlay">
  <div class="proc-card">
    <div class="proc-card__status">
      <span class="proc-card__dot"></span>
      <span class="proc-card__label">{stage_label}</span>
    </div>
    <div class="proc-card__video">
      <video id="em-loader-video" autoplay loop muted playsinline>
        <source src="data:video/mp4;base64,__EM_LOADER_VIDEO_B64__" type="video/mp4">
      </video>
      <div class="proc-card__vignette"></div>
      <div class="proc-card__live"><span class="proc-card__live-dot"></span>LIVE TRACE</div>
    </div>
    <div class="proc-card__caption">FlyWire / CAVE segmentation viewer</div>
    <div class="proc-card__hint">This might take a second&hellip; or two&hellip;</div>
  </div>
</div>
<script>
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const v = document.getElementById('em-loader-video');
    if (v) v.pause();
  }
</script>
""".replace("__EM_LOADER_VIDEO_B64__", EM_LOADER_VIDEO_B64)


def render_loader(stage_label: str) -> str:
    return _EM_LOADER_TEMPLATE.replace("{stage_label}", stage_label)


def render_setup_controls(use_columns: bool):
    """Caption + About panel + uploader + control-track picker + Run button.

    Shared by the full pre-run page and the compact "Change input" popover --
    `use_columns` just decides whether the track picker/button sit side by side
    (roomy pre-run layout) or stacked (narrow popover).
    """
    st.caption(
        "Real FlyWire connectome (~23k neurons, ~76k real synaptic connections). "
        "The wiring is real; the audio-to-activation mapping is a speculative "
        "simulation layer, not a validated model of real fly behavior."
    )
    with st.expander("About this project — what's real vs. speculative"):
        st.markdown(ABOUT_MD)

    uploaded = st.file_uploader("Drop an audio file", type=["mp3", "wav", "m4a", "flac", "ogg"])

    track_help = (
        "Defaults to a public-domain Chopin recording as a quick demo. White/pink noise "
        "are synthetic test signals, not music — useful for verifying the simulation "
        "reacts at all. Can sound harsh/loud; volume starts low."
    )
    processing = st.session_state.get("processing", False)
    button_label = "Running…" if processing else "Run simulation"

    if use_columns:
        col1, col2 = st.columns([1, 2])
        with col1:
            noise_choice = st.selectbox(
                "...or use a control track", list(CONTROL_TRACKS.keys()) + ["-"], help=track_help
            )
        with col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # align button with selectbox input
            run = st.button(button_label, type="primary", use_container_width=True, disabled=processing)
    else:
        noise_choice = st.selectbox(
            "...or use a control track", list(CONTROL_TRACKS.keys()) + ["-"], help=track_help
        )
        run = st.button(button_label, type="primary", use_container_width=True, disabled=processing)

    status_ph = st.empty()
    return uploaded, noise_choice, run, status_ph


has_result = "result" in st.session_state

if not has_result:
    # First visit: full setup takes the whole page, nothing to compete with yet.
    st.title("Fly Brain on Music")
    uploaded, noise_choice, run, status_ph = render_setup_controls(use_columns=True)
else:
    # A result already exists: the scene is the page now. Setup collapses into
    # one compact strip above it instead of pushing the scene below the fold.
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.markdown(
            "<div style='font-size:1.4rem; font-weight:700; margin-bottom:0.1rem;'>Fly Brain on Music</div>",
            unsafe_allow_html=True,
        )
    with top_col2:
        with st.popover("Change input", use_container_width=True):
            uploaded, noise_choice, run, status_ph = render_setup_controls(use_columns=False)

# Computing on button click, but RENDERING from session_state below -- st.button()
# only returns True on the exact script run where it was clicked, so any widget
# interaction afterward (e.g. the shell-style radio) triggers a rerun where `run`
# is False again. Gating the whole results section on `if run:` made the entire
# simulation output vanish the moment you touched a later widget. Persisting the
# computed result in session_state means later widget interactions just
# re-render from the stored result instead of wiping it.
if run:
    if uploaded is None and noise_choice == "-":
        status_ph.warning("Upload a file or pick a control track first.")
        st.stop()
    st.session_state["processing"] = True
    st.rerun()  # rerender once with the button disabled/labeled "Running…" before doing the work

if st.session_state.get("processing"):
    # If anything below raises (corrupt upload, decode error, etc.), "processing" must
    # still get reset -- otherwise the button is stuck disabled/"Running…" for the rest
    # of the session with no way to retry.
    try:
        if uploaded is not None:
            audio_bytes = uploaded.read()
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
            source_name = uploaded.name
        else:
            audio_bytes = (DATA_DIR / CONTROL_TRACKS[noise_choice]).read_bytes()
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
            source_name = noise_choice

        loader_ph = status_ph.empty()  # one slot, overwritten per stage -- a plain container() would stack both cards
        loader_ph.markdown(render_loader("Extracting audio features…"), unsafe_allow_html=True)
        features = extract_features(y, sr)
        loader_ph.markdown(render_loader("Propagating through the connectome…"), unsafe_allow_html=True)
        result = run_simulation(features, node_snapshot_stride=3)
    except Exception as e:
        st.session_state["processing"] = False
        status_ph.error(f"Couldn't process that track: {e}")
        st.stop()

    st.session_state["result"] = result
    st.session_state["features"] = features
    st.session_state["audio_bytes"] = audio_bytes
    st.session_state["source_name"] = source_name
    st.session_state["mime"] = mimetypes.guess_type(source_name)[0] or "audio/mpeg"
    st.session_state["processing"] = False
    st.rerun()  # immediately re-render in the compact "result exists" layout, scene included, no scroll needed

if "result" in st.session_state:
    result = st.session_state["result"]
    features = st.session_state["features"]
    audio_bytes = st.session_state["audio_bytes"]
    source_name = st.session_state["source_name"]
    mime = st.session_state["mime"]

    meta_col, shell_col = st.columns([1, 2])
    with meta_col:
        st.markdown(
            f"<div style='padding-top:0.4rem; color:#9a9a9a; font-size:0.85rem;'>"
            f"<b>{source_name}</b> — {features['tempo']:.0f} BPM, ~{features['times'][-1]:.0f}s</div>",
            unsafe_allow_html=True,
        )
    with shell_col:
        shell_choice = st.radio(
            "Brain/cord shell style",
            ["Smooth (density isosurface)", "Convex hull (faceted, stylistic)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    shell_style = "smooth" if shell_choice.startswith("Smooth") else "convex"
    with st.spinner("Building scene..."):
        html = build_scene_html(
            result["node_root_ids"], result["node_snapshots"], result["node_is_seed"], audio_bytes, mime,
            shell_style=shell_style,
        )
    st.markdown("<div style='margin-top:-0.8rem;'></div>", unsafe_allow_html=True)
    components.html(html, height=730)

    with st.expander("Simulation details — activation charts, frequency response, top neurons"):
        chart_df = pd.DataFrame(
            {
                "time_s": result["times"],
                "seed layer (Johnston's Organ)": result["seed"],
                "hop 1 (AMMC + direct)": result["hop1"],
                "hop 2+ (downstream)": result["hop2plus"],
            }
        ).set_index("time_s")
        st.subheader("Brain activation over time")
        st.line_chart(chart_df)

        st.subheader("Response by frequency band")
        st.caption(
            "Seed neurons are split into 3 groups, each driven mainly by bass/mid/treble energy "
            "(a modeling choice -- FlyWire's data has no real per-neuron frequency tuning "
            "annotation). Divergence between these lines means the sim is responding to "
            "spectral content, not just loudness."
        )
        band_df = pd.DataFrame(
            {
                "time_s": result["times"],
                "bass-group neurons": result["bass"],
                "mid-group neurons": result["mid"],
                "treble-group neurons": result["treble"],
            }
        ).set_index("time_s")
        st.line_chart(band_df)

        st.subheader("Most activated neurons at end of track")
        st.dataframe(pd.DataFrame(result["top_neurons"]), use_container_width=True)
