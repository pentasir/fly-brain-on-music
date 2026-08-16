"""A hand-authored schematic layout of fly brain + nerve cord neuropils, for
a lightweight 2D fallback visualization when the real 3D scene (web_scene.py,
brain_map_3d.py -- actual CAVE-derived neuron positions) isn't wanted or
available. Coordinates here are NOT anatomically precise measurements -- they
are a topologically-reasonable schematic: optic lobes flanking a central
brain, mushroom body/central-complex neuropils clustered above a ventral
gnathal ganglion, thoracic/abdominal nerve-cord neuromeres chained below it.
Good enough to show *where* activity concentrates region-by-region, not for
anatomical measurement.

Left/right hemisphere data is already merged in this project's region labels
(data/connectome/auditory_neurons.csv's "region" field doesn't carry L/R), so
each bubble represents both sides combined -- placed at the midline.

Region abbreviations follow standard FlyWire/BANC neuropil nomenclature.
Coordinates: x = left(-) to right(+), y = dorsal/brain(+) to ventral/tail(-).
"""
from collections import Counter

REGION_COORDS = {
    # --- optic lobes (paired, but our data is L/R-merged -> placed at midline x) ---
    "ME": (0.0, 5.4),      # medulla
    "LO": (0.0, 4.9),      # lobula
    "LOP": (0.0, 4.5),     # lobula plate
    "AME": (0.0, 5.7),     # accessory medulla
    "AOTU": (2.6, 4.2),    # anterior optic tubercle

    # --- superior/dorsal central brain ---
    "SLP": (-0.6, 4.6),    # superior lateral protocerebrum
    "SMP": (0.0, 4.7),     # superior medial protocerebrum
    "SIP": (0.4, 4.4),     # superior intermediate protocerebrum
    "SPS": (0.7, 4.0),     # superior posterior slope

    # --- mushroom body (learning/memory) ---
    "MB_CA": (1.6, 4.8),   # calyx
    "MB_PED": (1.3, 4.2),  # peduncle
    "MB_VL": (1.5, 5.1),   # vertical lobe
    "MB_ML": (1.1, 4.0),   # medial lobe

    # --- lateral / ventrolateral protocerebrum ---
    "LH": (2.2, 4.5),      # lateral horn
    "AVLP": (2.4, 3.6),    # anterior ventrolateral protocerebrum
    "PVLP": (2.6, 3.3),    # posterior ventrolateral protocerebrum
    "PLP": (2.0, 3.1),     # posterior lateral protocerebrum
    "WED": (2.8, 3.8),     # wedge

    # --- central complex ---
    "FB": (0.0, 3.7),      # fan-shaped body
    "EB": (0.0, 3.5),      # ellipsoid body
    "PB": (0.0, 3.9),      # protocerebral bridge
    "NO": (0.2, 3.6),      # noduli
    "NO_CONS": (0.2, 3.55),
    "BU": (-0.5, 3.6),     # bulb

    # --- inferior/ventral central brain ---
    "CRE": (0.0, 3.2),     # crepine
    "ICL": (0.5, 3.0),     # inferior clamp
    "IB": (-0.4, 3.1),     # inferior bridge
    "ATL": (-0.2, 2.9),    # antler
    "GOR": (0.3, 2.8),     # gorget
    "EPA": (0.6, 2.7),     # epaulette
    "IPS": (0.8, 2.6),     # inferior posterior slope
    "SCL": (-0.6, 2.9),    # superior clamp
    "LAL": (-1.0, 3.0),    # lateral accessory lobe

    # --- sensory input hubs ---
    "AL": (-1.3, 3.5),     # antennal lobe
    "AMMC": (-1.6, 2.6),   # antennal mechanosensory & motor center -- Johnston's Organ target
    "AMNP": (-1.8, 2.3),   # antennal mechanosensory neuropil (adjacent)

    # --- ventral brain / gnathal ---
    "SAD": (0.0, 1.9),     # saddle
    "FLA": (0.3, 1.8),     # flange
    "PRW": (-0.3, 1.7),    # prow
    "GNG": (0.0, 1.3),     # gnathal ganglion -- base of brain, gateway to VNC

    # --- ventral nerve cord: cervical -> thoracic -> abdominal, chained below ---
    "INTTCT": (0.0, 0.6),   # intermediate tectulum (neck)
    "NTCT": (0.0, 0.2),     # neck tectulum
    "HTCT": (0.3, -0.1),    # haltere tectulum
    "HTCT_UTCT_T3": (0.3, -0.15),
    "WTCT": (-0.3, -0.1),   # wing tectulum
    "LTCT": (0.0, -0.4),    # leg tectulum
    "T1_PRONM": (0.0, -0.8),    # prothoracic neuromere
    "T2_MESONM": (0.0, -1.4),   # mesothoracic neuromere
    "T2_MVAC": (0.3, -1.4),
    "T3_METANM": (0.0, -2.0),   # metathoracic neuromere
    "VES": (0.0, -2.4),         # vestibule
    "ABDNM": (0.0, -3.0),       # abdominal neuromere
}

FALLBACK_COORD = (0.0, 0.0)  # any region not in the table above (shouldn't normally happen)


def region_bubbles(node_regions, node_values):
    """Aggregate per-neuron activation into one (x, y, mean_value, count) bubble
    per region -- the schematic shows region-level activation, not individual
    neurons (there's no anatomically meaningful per-neuron position here)."""
    sums = Counter()
    counts = Counter()
    for region, value in zip(node_regions, node_values):
        sums[region] += float(value)
        counts[region] += 1

    bubbles = []
    for region, count in counts.items():
        x, y = REGION_COORDS.get(region, FALLBACK_COORD)
        bubbles.append(
            {"region": region, "x": x, "y": y, "mean_activation": sums[region] / count, "n_neurons": count}
        )
    return bubbles


def build_schematic_figure(node_regions, node_values, cmin=None, cmax=None):
    """Static matplotlib fallback: one bubble per region, sized by neuron count,
    colored by mean activation. Use when the interactive 3D scene (web_scene.py)
    isn't wanted/available -- same underlying activation data, coarser view."""
    import matplotlib.pyplot as plt

    bubbles = region_bubbles(node_regions, node_values)
    values = [b["mean_activation"] for b in bubbles]
    if cmin is None:
        cmin = min(values) if values else 0.0
    if cmax is None:
        cmax = max(values) if values else 1.0
    if cmax - cmin < 1e-6:
        cmax = cmin + 1.0

    fig, ax = plt.subplots(figsize=(6, 8), facecolor="#03030a")
    ax.set_facecolor("#03030a")

    xs = [b["x"] for b in bubbles]
    ys = [b["y"] for b in bubbles]
    sizes = [30 + 4 * (b["n_neurons"] ** 0.5) for b in bubbles]

    sc = ax.scatter(xs, ys, s=sizes, c=values, cmap="inferno", vmin=cmin, vmax=cmax, edgecolors="#8f5ad6", linewidths=0.5)
    for b in bubbles:
        ax.annotate(b["region"], (b["x"], b["y"]), fontsize=6, color="#999", xytext=(0, 6), textcoords="offset points", ha="center")

    ax.set_title("Fly brain + nerve cord (schematic)", color="#ddd", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(sc, ax=ax, label="mean activation", fraction=0.04)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Quick smoke test with synthetic values -- one bar of activation per
    # region so REGION_COORDS coverage is easy to eyeball.
    import random

    regions = list(REGION_COORDS.keys())
    node_regions = regions
    node_values = [random.random() for _ in regions]
    fig = build_schematic_figure(node_regions, node_values)
    fig.savefig("brain_map_schematic_preview.png", dpi=150)
    print("wrote brain_map_schematic_preview.png")
