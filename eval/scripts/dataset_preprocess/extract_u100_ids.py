import os
import polars as pl


def extract_u100_ids():
    base_path = os.path.join(os.environ.get("DATA_DIR", "/path/to/data"), "uniref/train")

    for file in os.listdir(base_path):
        if "uniref100" in file and file.endswith(".csv"):
            input_path = os.path.join(base_path, file)
            output_file = os.path.join(base_path, f"{os.path.splitext(file)[0]}_ids.csv")

            df_lazy = pl.scan_csv(input_path).select(pl.col("name").str.replace("UniRef100_", ""))
            df_lazy.sink_csv(output_file)
            print(f"Extracted ids from {file} and saved them to {output_file}")


def match_with_evaluation_datasets():

    base_path = os.path.join(os.environ.get("DATA_DIR", "/path/to/data"), "evaluation/")
    train_ids = os.path.join(os.environ.get("DATA_DIR", "/path/to/data"), "uniref/train/uniref100_ids.csv")

    # Collect the list of names from the train set
    train_ids_df = pl.read_csv(train_ids, has_header=False)
    train_ids_df = train_ids_df.rename({train_ids_df.columns[0]: "name"})
    train_ids_list = set(train_ids_df["name"].to_list())

    for file in os.listdir(base_path):
        if file.endswith("proteome.csv"):
            path = os.path.join(base_path, file)
            df = pl.read_csv(path).drop("sequence")

            # Extract Uniprot ID and check for membership
            df = df.with_columns([pl.col("name").str.split("|").list.get(1).alias("name")]).with_columns(
                [pl.col("name").is_in(train_ids_list).alias("in_train")]
            )

            output_path = path.replace(".csv", "_in_train.csv")
            df.write_csv(output_path)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    # extract_u100_ids()
    match_with_evaluation_datasets()
