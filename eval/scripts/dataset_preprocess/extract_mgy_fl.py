import os
import pickle
import pandas as pd
import polars as pl
from tqdm import tqdm


if __name__ == "__main__":
    base_path = os.path.join(os.environ.get("SCORES_DIR", "/path/to/scores"), "scores")
    data_path = os.environ.get("EVAL_DATA_DIR", "/path/to/AMPLIFY-evaluation/data")
    # file_name = "mgy_1M_scores.csv"
    file_name = "mgnify_train_all_ambig.csv"

    df = pl.read_csv(os.path.join(data_path, "mgy_1M_scores.csv"))
    df = df.with_columns(pl.lit(None).alias("FL"))

    dict_chunks_dir = os.path.join(os.environ.get("DATA_DIR", "/path/to/data"), "mgnify/download")
    dict_files = sorted(f for f in os.listdir(dict_chunks_dir) if f.endswith("_fl.pkl"))

    # Iteratively join each dict chunk
    for i, dict_file in tqdm(enumerate(dict_files), total=len(dict_files)):
        # Load dict chunk
        with open(os.path.join(dict_chunks_dir, dict_file), "rb") as f:
            chunk_dict = pickle.load(f)

        # Convert to Polars LazyFrame
        dict_chunk_lf = pl.DataFrame({"name": list(chunk_dict.keys()), "chunk_bool": list(chunk_dict.values())})

        # Left join and fill nulls in 'name_bool' only where missing
        df = df.join(dict_chunk_lf, on="name", how="left").with_columns(pl.col("FL").fill_null(pl.col("chunk_bool"))).drop("chunk_bool")

    df.write_csv(os.path.join(data_path, file_name.replace(".csv", "_fl.csv")))
