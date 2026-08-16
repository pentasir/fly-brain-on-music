"""
Build the auditory-pathway subgraph from the FlyWire/Codex public data export.

Seed set: neurons on the left/right antennal nerve (Johnston's Organ afferents).
Expands N_HOPS downstream through the real synaptic connectivity, keeping only
edges with at least MIN_SYN synapses (raw connectome is dense; without a
threshold a 2-hop expansion from ~4,500 seed neurons balloons toward most of
the brain).

Inputs:  data/connectome/neurons.csv.gz, data/connectome/connections_princeton.csv.gz
Outputs: data/connectome/auditory_graph.gpickle
         data/connectome/auditory_neurons.csv   (node metadata for everything kept)
"""
import gzip
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"
N_HOPS = 3
MIN_SYN = 12

SEED_NERVES = {"left_antennal_nerve", "right_antennal_nerve"}


def load_neurons() -> pd.DataFrame:
    with gzip.open(DATA_DIR / "neurons.csv.gz", "rt") as f:
        df = pd.read_csv(f, dtype={"Root ID": "int64"}, low_memory=False)
    return df.set_index("Root ID", drop=False)


def load_connections() -> pd.DataFrame:
    with gzip.open(DATA_DIR / "connections_princeton.csv.gz", "rt") as f:
        df = pd.read_csv(
            f,
            dtype={"pre_root_id": "int64", "post_root_id": "int64", "syn_count": "int32"},
        )
    return df[df["syn_count"] >= MIN_SYN]


def expand_downstream(seed_ids: set, connections: pd.DataFrame, n_hops: int):
    """BFS outward along pre -> post edges, hop by hop."""
    visited = set(seed_ids)
    frontier = set(seed_ids)
    kept_edges = []

    for hop in range(n_hops):
        hop_edges = connections[connections["pre_root_id"].isin(frontier)]
        kept_edges.append(hop_edges)
        next_frontier = set(hop_edges["post_root_id"]) - visited
        print(f"  hop {hop + 1}: {len(frontier)} sources -> {len(hop_edges)} edges -> {len(next_frontier)} new neurons")
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return visited, pd.concat(kept_edges, ignore_index=True) if kept_edges else connections.iloc[0:0]


def main():
    print("Loading neurons...")
    neurons = load_neurons()

    seed = neurons[neurons["Nerve"].isin(SEED_NERVES)]
    seed_ids = set(seed["Root ID"])
    print(f"Seed set (Johnston's Organ / antennal nerve afferents): {len(seed_ids)} neurons")

    print("Loading connections (this is the big file)...")
    connections = load_connections()
    print(f"Connections with syn_count >= {MIN_SYN}: {len(connections)}")

    print(f"Expanding {N_HOPS} hops downstream...")
    all_ids, edges = expand_downstream(seed_ids, connections, N_HOPS)
    print(f"Total neurons in subgraph: {len(all_ids)}")
    print(f"Total edges in subgraph: {len(edges)}")

    print("Building graph...")
    graph = nx.DiGraph()
    for nid in all_ids:
        is_seed = nid in seed_ids
        row = neurons.loc[nid] if nid in neurons.index else None
        region = None
        if row is not None:
            raw_region = row["Top in/out region"]
            if isinstance(raw_region, str) and raw_region:
                region = raw_region.split(".")[0]  # compound labels like "ME.LO" -> primary region
        graph.add_node(
            nid,
            is_seed=is_seed,
            cell_class=row["Class"] if row is not None else None,
            cell_type=row["Primary Cell Type"] if row is not None else None,
            nerve=row["Nerve"] if row is not None else None,
            region=region,
        )
    for _, e in edges.iterrows():
        if e["pre_root_id"] in graph and e["post_root_id"] in graph:
            graph.add_edge(e["pre_root_id"], e["post_root_id"], weight=int(e["syn_count"]), neuropil=e["neuropil"])

    out_graph = DATA_DIR / "auditory_graph.gpickle"
    with open(out_graph, "wb") as f:
        pickle.dump(graph, f)
    print(f"Saved graph: {out_graph} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")

    node_rows = []
    for nid, attrs in graph.nodes(data=True):
        node_rows.append({"root_id": nid, **attrs})
    out_csv = DATA_DIR / "auditory_neurons.csv"
    pd.DataFrame(node_rows).to_csv(out_csv, index=False)
    print(f"Saved node metadata: {out_csv}")


if __name__ == "__main__":
    main()
