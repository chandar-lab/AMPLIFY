import csv
import os
from argparse import ArgumentParser

# from cuml import UMAP
from umap import UMAP
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import time


def load_df(score_path, sequences_path):
    df = []
    for file in os.listdir(score_path):
        if file.endswith(".parquet"):
            df.append(pd.read_parquet(os.path.join(score_path, file)))

    output = pd.concat(df)

    with open(sequences_path, "rt") as f:
        output_sequences = pd.read_csv(f)

    output = output.merge(output_sequences, left_on="name", right_on="name", how="inner")
    output["length"] = output["sequence"].apply(lambda x: len(x))

    return output


def project_df(df, reducer, embedding_dim=2048):
    output = {}

    for _, row in df.iterrows():
        token_embbedings = []
        for i in range(embedding_dim):
            token_emb = row.get(f"embedding_{i}")
            if token_emb is not None:
                token_embbedings.append(token_emb)
        if token_embbedings:
            token_projections = reducer.transform(token_embbedings)
            output[row.get("name")] = {
                "pseudo_ppl": row.get("pseudo_ppl"),
                "mean_cosine_sim": row.get("mean_cosine_sim"),
                "std_cosine_sim": row.get("std_cosine_sim"),
                "token_projections": token_projections,
            }

    return output


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--score_path", type=str, default="/path/to/scores/Embedding-scores/u100_all_layers")
    args.add_argument("--sequences_path", type=str, default="/path/to/data/uniref100_train_10k.csv")
    args.add_argument("--output_path")
    args.add_argument("--n_samples", type=int, default=1000000)
    args = args.parse_args()

    df = load_df(
        args.score_path,
        args.sequences_path,
    )

    os.makedirs(args.output_path, exist_ok=True)

    # Collect all token embeddings and metadata
    all_embeddings = []
    text_residue = []
    positions = []
    ids = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        token_emb = row.get(f"token_embeddings")[1:-1]  # We skip the BOS and EOS token
        sequence = list(row["sequence"][: len(token_emb)])
        all_embeddings.extend(token_emb)
        text_residue.extend(sequence)
        positions.extend(list(range(len(token_emb))))
        ids.extend([row["name"]] * len(token_emb))

    idx = np.random.choice(len(all_embeddings), size=args.n_samples, replace=False)
    X_sampled = np.array(all_embeddings)[idx]

    matched_residue = np.array(text_residue)[idx]
    matched_position = np.array(positions)[idx]
    matched_id = np.array(ids)[idx]
    np.save(os.path.join(args.output_path, "matched_residue.npy"), matched_residue)
    np.save(os.path.join(args.output_path, "matched_position.npy"), matched_position)
    np.save(os.path.join(args.output_path, "matched_id.npy"), matched_id)

    del text_residue, all_embeddings, matched_residue

    # Fit UMAP
    start_time = time.time()
    umap_model = UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    X_umap = umap_model.fit_transform(X_sampled)
    # umap_model = UMAP(
    #     n_neighbors=128,  # Balances local vs. global structure.
    #     n_components=2,  # Target dimension for visualization.
    #     min_dist=0.1,  # Controls how tightly points are packed.
    #     n_epochs=1000,  # Number of optimization iterations.
    #     build_algo="nn_descent",  # Use the fast approximate nearest neighbor algorithm on the GPU.
    #     # Low-level keywords to fine-tune the nn_descent algorithm for performance and memory.
    #     build_kwds={
    #         "nnd_do_batch": True,
    #         "nnd_n_clusters": 8,
    #         "nnd_graph_degree": 256,
    #         "nnd_intermediate_graph_degree": 512,
    #         "nnd_overlap_factor": 7,
    #     },
    # )
    # X_umap = umap_model.fit_transform(X_sampled, data_on_host=True)
    end_time = time.time()
    print(f"Project the data in 2D with UMAP ({end_time - start_time:.2f}s)")

    joblib.dump(umap_model, os.path.join(args.output_path, "umap_reducer.joblib"))
    np.save(os.path.join(args.output_path, "X_umap.npy"), X_umap)

    # reducer = joblib.load("/path/to/visuals/umap_reducer.joblib")

    # bad_proteins = project_df(pe5[(pe5["length"] >= 90) & (pe5["length"] < 120) & (pe5["pseudo_ppl"] > 3.5)], reducer)
    # good_proteins = project_df(df[(df["length"] >= 90) & (df["length"] < 120) & (df["pseudo_ppl"] < 1)], reducer)

    # with open("/path/to/visuals/bad_proteins.pkl", "wb") as f:
    #     pickle.dump(bad_proteins, f)

    # with open("/path/to/visuals/good_proteins.pkl", "wb") as f:
    #     pickle.dump(good_proteins, f)
