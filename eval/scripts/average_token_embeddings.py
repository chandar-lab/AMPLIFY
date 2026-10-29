import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def process_file(path):
    df = pd.read_parquet(path)

    emb = [np.asarray(np.stack(x), dtype=np.float32).mean(axis=0).astype(np.float16) for x in df["token_embeddings_per_layer"]]  #
    df = df.drop(columns=["token_embeddings_per_layer"])
    df["embedding"] = emb

    out_file = os.path.join(out_path, os.path.basename(path))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_parquet(out_file, engine="pyarrow")


base_path = os.path.join(os.environ.get("SCORES_DIR", "/path/to/scores"), "Embedding-scores/biomap_research/token_level")
out_path = os.path.join(os.environ.get("SCORES_DIR", "/path/to/scores"), "Embedding-scores/biomap_research/mean")

os.makedirs(out_path, exist_ok=True)

files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".parquet")]

# for file in files:
#     process_file(file)

max_workers = os.cpu_count() - 1 or 8
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    list(tqdm(ex.map(process_file, files), total=len(files)))
