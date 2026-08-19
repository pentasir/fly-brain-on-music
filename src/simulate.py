"""Drive the real FlyWire auditory-pathway graph with audio features.

Model: leaky integrate-and-fire (LIF) neurons, Euler-integrated, connected
by real synaptic weights (LINEAR in raw synapse count, see W_SYN_MV below)
SIGNED by each presynaptic neuron's real predicted neurotransmitter (see
NT_SIGN below). Per neuron i, membrane potential V_i evolves in real mV
units as:
    tau_m * dV_i/dt = -(V_i - V_rest) + I_syn_i(t) + I_drive_i(t)
and V_i spikes (V_i -> V_reset, enters refractory) when it crosses V_th.
I_syn_i(t) is the signed synaptic current from presynaptic spikes in the
previous timestep -- real spiking, real inhibition, real refractory
nonlinearity, not a generic tanh squashing function.

Parameters (tau_m, V_rest, V_th, T_refrac, W_syn) are taken from Shiu et al.
2024, Nature, "A Drosophila computational brain model reveals sensorimotor
processing" -- a leaky integrate-and-fire model built on this SAME FlyWire
connectome. Their tau_m (= R_mbr * C_mbr = 10 MOhm * 0.002 uF = 20 ms)
independently matches Gouwens & Wilson 2009's direct electrophysiological
measurement (also ~20 ms), a genuine cross-validation between two different
sources, not a coincidence of us picking the same round number twice.
W_syn = 0.275 mV is their model's single free (fitted) parameter -- the
membrane-potential jump per synapse, per presynaptic spike -- and the
synapse-count-to-weight relationship is linear (raw count * sign * W_syn),
not the log-compressed, unitless weighting used in earlier versions of this
model. I_drive (the audio-to-current mapping) has no literature source --
that part remains our own artistic modeling layer, clearly distinguished
from the cited synaptic/membrane parameters above.

Resolution trade-off, stated plainly: integrating at the real ~1 ms
timescale for a multi-minute track across ~26k neurons is not tractable for
an interactive local app. Each audio-derived frame (spanning roughly
100ms-2s depending on track length) is instead substepped internally
(N_SUBSTEPS Euler steps per frame, dt sized so tau_m is resolved reasonably
well within a frame) -- the ODE and its real cited time constant are
genuine, but audio-driven input is held constant across the substeps within
one frame (a coarse-graining assumption, not claimed to be biologically
exact at sub-frame timescales).

This is still a deliberately simple, speculative model -- it is not a
validated simulation of real fly neural dynamics (no dataset of real fly
neural response to music exists to validate against), but it is now a
genuine spiking LIF simulation with a real, cited membrane time constant,
not an arbitrary recurrence formula.

Neurotransmitter sign convention (from `neurons.csv`, a per-neuron field --
transmitter identity is a property of the presynaptic neuron, not of the
individual synapse). "Verified NT type" (higher confidence, smaller
coverage) is preferred where present; falls back to "Predicted NT type"
otherwise -- see VERIFIED_NAME_TO_CODE / _load_nt_types:
    ACH (acetylcholine) -> excitatory. Nicotinic ACh receptors in the fly
        CNS are cation channels -- the primary fast excitatory transmitter.
    GABA -> inhibitory. GABA-A / GluCl-type receptors, well established.
    GLUT (glutamate) -> inhibitory. Unlike vertebrate CNS, most glutamatergic
        synapses in Drosophila act through glutamate-gated chloride channels
        (GluClalpha) and are inhibitory, not excitatory. This convention is
        used in published connectome-constrained models, e.g. Lappalainen et
        al. 2024, Nature, "Connectome-constrained networks predict neural
        activity across the fly visual system."
    HIST (histamine) -> inhibitory. Histamine-gated chloride channels are
        the photoreceptor->L1/L2 synapse mechanism (Hardie, 1989, Nature,
        "A histamine-activated chloride channel involved in neurotransmission
        at a photoreceptor synapse").
    DA/SER/OCT/TYR (dopamine, serotonin, octopamine, tyramine) -> biogenic
        amine neuromodulators with both excitatory and inhibitory receptor
        subtypes depending on target; no single universal sign in the
        literature, so given a small, roughly-neutral magnitude rather than
        a confident sign.
    Unresolved (no prediction) -> same small neutral magnitude.

`drive_vec` is NOT a single scalar broadcast to every seed neuron -- the seed
population is split N_BANDS ways (band_group, see _build_arrays) and each
group is driven mainly by its own mel-spaced frequency band (low -> high,
see audio_features.py), plus a shared rhythm/loudness term everyone feels.
This is a deterministic, documented modeling choice, not a claim that
FlyWire's public data has real per-neuron frequency-tuning annotations (it
doesn't).
"""
import gzip
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from audio_features import N_BANDS

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"

NT_SIGN = {
    "ACH": 1.0,
    "GABA": -1.0,
    "GLUT": -1.0,
    "HIST": -1.0,
    "DA": 0.3,
    "SER": 0.3,
    "OCT": 0.3,
    "TYR": 0.3,
}
NT_SIGN_DEFAULT = 0.3  # unresolved prediction -- same small neutral magnitude as the amine modulators

# neurons.csv's "Verified NT type" spells transmitters out in full (sometimes as a
# comma-separated multi-transmitter list, e.g. "gaba,nitric_oxide" for a neuron
# co-releasing both) -- map to the same short codes used by "Predicted NT type".
# For multi-transmitter entries, the FIRST listed transmitter is taken as primary
# (a simplification: real co-release can't be represented by a single sign here).
VERIFIED_NAME_TO_CODE = {
    "acetylcholine": "ACH",
    "gaba": "GABA",
    "glutamate": "GLUT",
    "histamine": "HIST",
    "dopamine": "DA",
    "serotonin": "SER",
    "octopamine": "OCT",
    "tyramine": "TYR",
    # no short-code equivalent (nitric oxide, glycine) -- falls through to NT_SIGN_DEFAULT
}

_cache = {}


def _load_nt_types() -> dict:
    if "nt_types" not in _cache:
        with gzip.open(DATA_DIR / "neurons.csv.gz", "rt") as f:
            df = pd.read_csv(
                f, usecols=["Root ID", "Predicted NT type", "Verified NT type"], dtype={"Root ID": "int64"}
            )

        def resolve(row):
            # Prefer verified (higher confidence, smaller coverage) over predicted.
            verified = row["Verified NT type"]
            if isinstance(verified, str) and verified:
                primary = verified.split(",")[0].strip().lower()
                code = VERIFIED_NAME_TO_CODE.get(primary)
                if code:
                    return code
            return row["Predicted NT type"]

        df["resolved_nt"] = df.apply(resolve, axis=1)
        _cache["nt_types"] = dict(zip(df["Root ID"], df["resolved_nt"]))
    return _cache["nt_types"]


def _load_graph():
    if "graph" not in _cache:
        with open(DATA_DIR / "auditory_graph.gpickle", "rb") as f:
            _cache["graph"] = pickle.load(f)
    return _cache["graph"]


def _build_arrays():
    if "arrays" in _cache:
        return _cache["arrays"]

    graph = _load_graph()
    nodes = list(graph.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    seed_mask = np.zeros(n, dtype=np.float64)
    for node, attrs in graph.nodes(data=True):
        if attrs.get("is_seed"):
            seed_mask[index[node]] = 1.0

    # Tonotopic-style split: FlyWire's public export has no per-neuron frequency
    # tuning annotation, so this is NOT a real tonotopic map -- it's a deterministic
    # partition of the seed population into N_BANDS groups (by root_id, stable
    # across runs), each driven primarily by a different mel-spaced frequency band
    # (low -> high). Modeling choice, not a claim about real per-neuron tuning.
    band_group = np.full(n, -1, dtype=np.int8)  # -1 = not a seed neuron
    for node, attrs in graph.nodes(data=True):
        if attrs.get("is_seed"):
            band_group[index[node]] = node % N_BANDS

    nt_types = _load_nt_types()
    nt_sign = np.array([NT_SIGN.get(nt_types.get(node), NT_SIGN_DEFAULT) for node in nodes])

    # Linear synapse-count-to-weight conversion, matching Shiu et al. 2024 (Nature,
    # "A Drosophila computational brain model reveals sensorimotor processing" --
    # a leaky integrate-and-fire model built on this same FlyWire connectome):
    # weight_ij = raw_synapse_count_ij * sign_i * W_SYN_MV, where W_SYN_MV=0.275mV
    # is their fitted single free parameter (mV added to the downstream membrane
    # potential per synapse, per presynaptic spike). This replaces an earlier
    # log1p-normalized, unitless weight with a literature-sourced linear one.
    rows, cols, weights = [], [], []
    for u, v, attrs in graph.edges(data=True):
        rows.append(index[u])
        cols.append(index[v])
        weights.append(attrs.get("weight", 1) * nt_sign[index[u]] * W_SYN_MV)
    A = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))

    # multi-source BFS from all seed nodes -> hop distance per neuron
    dist = np.full(n, -1, dtype=int)
    q = deque()
    for node, attrs in graph.nodes(data=True):
        if attrs.get("is_seed"):
            dist[index[node]] = 0
            q.append(node)
    while q:
        u = q.popleft()
        du = dist[index[u]]
        for v in graph.successors(u):
            vi = index[v]
            if dist[vi] == -1:
                dist[vi] = du + 1
                q.append(v)

    labels = np.array(
        [graph.nodes[n].get("cell_type") or graph.nodes[n].get("cell_class") or "unknown" for n in nodes],
        dtype=object,
    )
    root_ids = np.array(nodes)
    regions = np.array([graph.nodes[n].get("region") or "unknown" for n in nodes], dtype=object)

    _cache["arrays"] = dict(
        A=A, seed_mask=seed_mask, dist=dist, labels=labels, root_ids=root_ids, regions=regions, band_group=band_group
    )
    return _cache["arrays"]


MIN_REGION_SIZE = 15  # ignore regions with too few neurons in the subgraph -- noisy means

# Shiu et al. 2024 (Nature) parameters -- see module docstring for sourcing
TAU_M_MS = 20.0  # = R_mbr(10 MOhm) * C_mbr(0.002 uF); independently matches Gouwens & Wilson 2009
V_REST = -52.0  # mV
V_TH = -45.0  # mV -- 7 mV above rest
V_RESET = -52.0  # mV, resets to rest
T_REFRAC_MS = 2.2
W_SYN_MV = 0.275  # mV per synapse, per presynaptic spike -- Shiu et al.'s single fitted free parameter
# Euler integration is only numerically stable while dt stays well below tau_m
# (dt/tau ~ 0.25 or smaller is a safe margin -- larger and each step overshoots,
# causing oscillation that can average out to near-zero or even flip sign in the
# per-frame reported value, which looked like "nothing lights up" for real songs:
# a fixed substep COUNT (not a fixed dt) meant longer tracks -- with longer audio
# frames -- silently got a much larger, unstable dt). Target dt directly instead.
TARGET_DT_MS = 5.0
MAX_SUBSTEPS_PER_FRAME = 800  # bounds worst-case runtime for unusually long tracks


def run_simulation(
    features: dict,
    alpha: float = 1.0,  # matches Shiu et al. exactly (no extra scaling beyond W_syn) when left at 1.0
    drive_scale: float = 12.0,  # mV -- our own artistic parameter (no literature source), sized against the 7mV rest-to-threshold gap
    node_snapshot_stride: int = 0,
) -> dict:
    arrays = _build_arrays()
    A = arrays["A"]
    seed_mask = arrays["seed_mask"]
    dist = arrays["dist"]
    regions = arrays["regions"]
    band_group = arrays["band_group"]
    n = len(seed_mask)
    n_frames = len(features["times"])

    # shared rhythm/loudness component (all bands feel the beat), plus a
    # band-specific component (each band-group's neurons respond mainly to their
    # own mel-spaced frequency band, and get an extra kick from transients
    # detected specifically within that band, e.g. a kick drum vs. a cymbal hit
    # firing their respective bands independently rather than both waiting on
    # the whole-mix onset envelope) -- this is what makes the sim differentiate
    # spectral content, not just overall loudness.
    shared = 0.15 * features["rms"] + 0.1 * features["onset"]
    drive_by_band = {
        b: drive_scale * (shared + 0.75 * features["bands"][b] + 0.4 * features["band_onsets"][b])
        for b in range(N_BANDS)
    }
    band_masks = {b: (band_group == b).astype(np.float64) for b in range(N_BANDS)}

    V = np.full(n, V_REST, dtype=np.float64)
    refrac = np.zeros(n, dtype=np.float64)  # ms remaining in refractory period
    spike_prev = np.zeros(n, dtype=np.float64)  # binary spike train from the previous substep

    seed_idx = dist == 0
    hop1_idx = dist == 1
    hop2plus_idx = dist >= 2

    region_names, region_inverse, region_counts = np.unique(regions, return_inverse=True, return_counts=True)
    keep_regions = region_counts >= MIN_REGION_SIZE
    n_regions = len(region_names)

    seed_trace, hop1_trace, hop2_trace, total_trace = [], [], [], []
    band_traces = [[] for _ in range(N_BANDS)]
    region_series = np.zeros((n_frames, n_regions), dtype=np.float64)
    node_snapshots = []  # list of (time, state.copy()) if node_snapshot_stride > 0

    band_idx = [band_group == b for b in range(N_BANDS)]

    times = features["times"]
    frame_duration_s = (times[1] - times[0]) if n_frames > 1 else max(times[-1], 0.1)
    frame_duration_ms = frame_duration_s * 1000.0
    # Substep count derived from a FIXED target dt, not the other way around --
    # long tracks have long audio frames, and a fixed substep count would silently
    # blow dt past tau_m (numerically unstable) for anything much longer than the
    # ~30s clips this was originally tuned against. See constants above.
    n_substeps = int(np.clip(round(frame_duration_ms / TARGET_DT_MS), 1, MAX_SUBSTEPS_PER_FRAME))
    dt_ms = frame_duration_ms / n_substeps

    for t in range(n_frames):
        drive_vec = sum(drive_by_band[b][t] * band_masks[b] for b in range(N_BANDS))

        V_sum = np.zeros(n, dtype=np.float64)
        for _ in range(n_substeps):
            active = refrac <= 0
            I_syn = alpha * (A.T @ spike_prev)
            dV = (dt_ms / TAU_M_MS) * (-(V - V_REST) + I_syn + drive_vec)
            V = np.where(active, V + dV, V_RESET)
            spike = active & (V >= V_TH)
            V = np.where(spike, V_RESET, V)
            refrac = np.where(spike, T_REFRAC_MS, np.maximum(refrac - dt_ms, 0.0))
            spike_prev = spike.astype(np.float64)
            V_sum += V

        # frame-representative membrane potential, reported RELATIVE TO REST (not raw mV)
        # so downstream coloring/charts keep their existing "0 = rest, + = depolarized,
        # - = hyperpolarized below rest" semantics unchanged despite the real V_REST=-52mV.
        state = (V_sum / n_substeps) - V_REST

        seed_trace.append(float(state[seed_idx].mean()) if seed_idx.any() else 0.0)
        hop1_trace.append(float(state[hop1_idx].mean()) if hop1_idx.any() else 0.0)
        hop2_trace.append(float(state[hop2plus_idx].mean()) if hop2plus_idx.any() else 0.0)
        total_trace.append(float(state.mean()))
        for b in range(N_BANDS):
            band_traces[b].append(float(state[band_idx[b]].mean()) if band_idx[b].any() else 0.0)

        sums = np.bincount(region_inverse, weights=state, minlength=n_regions)
        region_series[t] = sums / np.maximum(region_counts, 1)

        if node_snapshot_stride and (t % node_snapshot_stride == 0 or t == n_frames - 1):
            node_snapshots.append((float(features["times"][t]), state.astype(np.float32).copy()))

    top_n = 15
    top_order = np.argsort(state)[::-1][:top_n]
    top_neurons = [
        {
            "root_id": int(arrays["root_ids"][i]),
            "label": str(arrays["labels"][i]),
            "hop": int(dist[i]),
            "activation": float(state[i]),
        }
        for i in top_order
    ]

    region_result = {
        name: region_series[:, i].tolist()
        for i, name in enumerate(region_names)
        if keep_regions[i] and name != "unknown"
    }
    region_final = {name: vals[-1] for name, vals in region_result.items()}
    region_counts_result = {
        name: int(region_counts[i]) for i, name in enumerate(region_names) if keep_regions[i] and name != "unknown"
    }

    return dict(
        times=features["times"],
        seed=seed_trace,
        hop1=hop1_trace,
        hop2plus=hop2_trace,
        total=total_trace,
        bands=band_traces,  # list of N_BANDS traces, low frequency -> high
        top_neurons=top_neurons,
        region_series=region_result,
        region_final=region_final,
        region_counts=region_counts_result,
        node_snapshots=node_snapshots,
        node_root_ids=arrays["root_ids"],
        node_labels=arrays["labels"],
        node_is_seed=seed_mask.astype(bool),
    )
