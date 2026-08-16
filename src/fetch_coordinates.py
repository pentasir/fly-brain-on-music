"""
Pull real 3D coordinates for every neuron in our subgraph from CAVE's
`cell_representative_point` table (datastack: brain_and_nerve_cord_public,
the published BANC release matching our downloaded CSVs -- NOT
flywire_fafb_production, which is a different, contributor-only dataset).

Cached locally so we don't re-query CAVE on every run.

Output: data/connectome/neuron_positions.csv (root_id, x_nm, y_nm, z_nm)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "connectome"
BATCH_SIZE = 500


def main():
    neurons = pd.read_csv(DATA_DIR / "auditory_neurons.csv")
    root_ids = neurons["root_id"].tolist()
    print(f"Fetching positions for {len(root_ids)} neurons...")

    client = CAVEclient("brain_and_nerve_cord_public")
    batches = np.array_split(root_ids, max(1, len(root_ids) // BATCH_SIZE))

    rows = []
    for i, batch in enumerate(batches):
        df = client.materialize.query_table(
            "cell_representative_point", filter_in_dict={"pt_root_id": list(batch)}
        )
        rows.append(df[["pt_root_id", "pt_position"]])
        if (i + 1) % 10 == 0 or i == len(batches) - 1:
            print(f"  batch {i + 1}/{len(batches)}")

    result = pd.concat(rows, ignore_index=True)
    result = result.rename(columns={"pt_root_id": "root_id"})
    coords = result["pt_position"].apply(pd.Series)
    coords.columns = ["x_nm", "y_nm", "z_nm"]
    out = pd.concat([result["root_id"], coords], axis=1)
    out = out.drop_duplicates(subset="root_id")

    print(f"Got positions for {len(out)} / {len(root_ids)} neurons ({100 * len(out) / len(root_ids):.1f}%)")

    out_path = DATA_DIR / "neuron_positions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
