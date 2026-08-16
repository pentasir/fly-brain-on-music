"""
Pull a random sample of neuron positions across the WHOLE brain+nerve cord
(not just our auditory subgraph) to use as a faint background silhouette --
without it, the subgraph alone looks like scattered dust rather than a
recognizable brain shape.

Output: data/connectome/context_positions.csv
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"
SAMPLE_SIZE = 20000
BATCH_SIZE = 500
SEED = 7


def main():
    with gzip.open(DATA_DIR / "neurons.csv.gz", "rt") as f:
        all_neurons = pd.read_csv(f, usecols=["Root ID"], dtype={"Root ID": "int64"})

    rng = np.random.default_rng(SEED)
    sample_ids = rng.choice(all_neurons["Root ID"].to_numpy(), size=SAMPLE_SIZE, replace=False)
    print(f"Sampling {SAMPLE_SIZE} neurons out of {len(all_neurons)} for background context...")

    client = CAVEclient("brain_and_nerve_cord_public")
    batches = np.array_split(sample_ids, SAMPLE_SIZE // BATCH_SIZE)

    rows = []
    for i, batch in enumerate(batches):
        df = client.materialize.query_table(
            "cell_representative_point", filter_in_dict={"pt_root_id": list(batch)}
        )
        rows.append(df[["pt_root_id", "pt_position"]])
        if (i + 1) % 10 == 0 or i == len(batches) - 1:
            print(f"  batch {i + 1}/{len(batches)}")

    result = pd.concat(rows, ignore_index=True).rename(columns={"pt_root_id": "root_id"})
    coords = result["pt_position"].apply(pd.Series)
    coords.columns = ["x_nm", "y_nm", "z_nm"]
    out = pd.concat([result["root_id"], coords], axis=1).drop_duplicates(subset="root_id")

    print(f"Got {len(out)} background positions")
    out_path = DATA_DIR / "context_positions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
