import os
import polars as pl

profiles_dir = os.path.join(os.environ.get("DATA_DIR", "/path/to/data"), "TRANSFER/PROFILES.50.50")
files = os.listdir(profiles_dir)

# Read all csv files into a list of DataFrames
dfs = []
for file in files:
    if file.endswith(".csv"):
        file_name = file.split(".")[0]
        df = pl.read_csv(os.path.join(profiles_dir, file))
        dfs.append((file_name, df))

# Concatenate all DataFrames into a single DataFrame with file names as a new column
df = pl.concat([df.with_columns(pl.lit(file_name).alias("id")) for file_name, df in dfs], how="vertical")

aa_cols = [col for col in df.columns if col not in ["id", "res_ix", "AA", ""]]

df = df.with_columns([pl.max_horizontal([pl.col(c) for c in aa_cols]).alias("max_freq")])
df = df.with_columns([pl.struct(aa_cols).alias("aa_freqs")])
def get_max_aa(struct_row):
    # struct_row is a dict: {"A": 0.1, "C": 0.0, ..., "Y": 0.0}
    return max(struct_row, key=struct_row.get)


df = df.with_columns([pl.col("aa_freqs").map_elements(get_max_aa, return_dtype=pl.String).alias("most_common")])

df = df.rename({"predicted_AA": "most_common"})

df = df.drop("aa_freqs")

def process_cat(df):
    df_target = df.filter(pl.col("AA") == pl.col("most_common"))
    df_non_target = df.filter(pl.col("AA") != pl.col("most_common"))

    # Target
    df_target = df_target.drop(col for col in df_target.columns if col not in ["id", "max_freq", "res_ix", "AA", "most_common"])
    df_target = df_target.with_columns(
        [(pl.col("res_ix") % 512).alias("res_ix_512"), (pl.col("res_ix") // 512).alias("number")]
    )
    df_target = df_target.cast({"number": pl.String})

    # Non target
    df_non_target = df_non_target.drop(
    col for col in df_non_target.columns if col not in ["id", "max_freq", "res_ix", "AA", "most_common"]
    )
    df_non_target = df_non_target.with_columns(
        [(pl.col("res_ix") % 512).alias("res_ix_512"), (pl.col("res_ix") // 512).alias("number")]
    )
    df_non_target = df_non_target.cast({"number": pl.String})
    
    df_target = df_target.rename({"res_ix": "res_ix_original", "res_ix_512": "res_ix"})
    df_non_target = df_non_target.rename({"res_ix": "res_ix_original", "res_ix_512": "res_ix"})

    return df_target, df_non_target

cat_90 = df.filter(pl.col("max_freq") >= 0.90)

cat_90_target, cat_90_non_target = process_cat(cat_90)

cat_90_target.write_parquet("cat_90_target.parquet")
cat_90_non_target.write_parquet("cat_90_non_target.parquet")

cat_96 = df.filter(pl.col("max_freq") >= 0.96)

cat_96_target, cat_96_non_target = process_cat(cat_96)

cat_96_target.write_parquet("cat_96_target.parquet")
cat_96_non_target.write_parquet("cat_96_non_target.parquet")

cat_70 = df.filter(pl.col("max_freq") > 0.70)

cat_70_target, cat_70_non_target = process_cat(cat_70)

cat_70_target.write_parquet("cat_70_target.parquet")
cat_70_non_target.write_parquet("cat_70_non_target.parquet")