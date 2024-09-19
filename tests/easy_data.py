import csv
import os

import pandas as pd
from xeger import Xeger

this_dir = os.path.dirname(__file__)


def regex_sequence_sampler(regex, max_len=32):
    """
    create a data generation function that samples from the space
    of strings matching the given regex.

    Args:
        n (int): the number of items to generate.
        out_path (PathLike, optional): path to which to persist the CSV result.
        max_len (int, optional, default=32): max length for a generated sequence.

    Returns: `pandas.DataFrame`
        in the same format as the PLM mask replacement task
        used by the model in this repo, with sequences sampled
        randomly from the space of regex matches.
    """

    def _generate_data(n, out_path=None):
        """
        see enclosing function doc string.

        Args:
          out_path (PathLike, optional): path to which to persist the CSV result.

        Returns: `pandas.DataFrame`
        """

        x = Xeger(limit=max_len)
        x.xeger(regex)
        fields = ["record_id", "record_name", "sequence_length", "sequence"]

        results = list()
        for k in range(n):
            seq = x.xeger(regex)
            results.append((k, f"thing_{k}", len(seq), seq))

        df = pd.DataFrame(results, columns=fields)

        if out_path is not None:
            df.to_csv(out_path)

        return df

    return _generate_data


"""
all data points are matching instances of this regular expression.
the choice made is fairly arbitrary, but is intended to produce
a set of synthetic sequences that shows some variation
but is patterned enough for the LLM to easily learn.
"""
unit_test_regex = "A{1,5}(D|C)(E|G)?(C|E|G)(E|G)VW(V|W)E?(S|T)(G|E)?(D|C)?(D|E)A{1,7}"
unit_test_distribution_sampler = regex_sequence_sampler(unit_test_regex)

"""
additionally, benchmark against sequences sampled from regular expressions
with varying degrees of similarity to the "true" distribution.
"""
out_of_sample_regex_trivial = "E{1,10}(C|D)F{7,13}"
out_of_sample_regex_easier = "(D|C)*(E|G)?(E|G)VW(V|W)E?(G|E)?(D|C)?(D|E){2,5}"
out_of_sample_regex_harder = "A{2,7}(D|C)(E|G)?(C|G)(E|G)(V|W)E?(S|T)(G|E)?(D|E)A{2,3}"

out_of_sample_regex_trivial_sampler = regex_sequence_sampler(
    out_of_sample_regex_trivial
)
out_of_sample_regex_easier_sampler = regex_sequence_sampler(out_of_sample_regex_easier)
out_of_sample_regex_harder_sampler = regex_sequence_sampler(out_of_sample_regex_harder)


if __name__ == "__main__":
    data = unit_test_distribution_sampler(
        5000000, out_path=os.path.join(this_dir, "..", "raw-test-data.csv")
    )

    df = pd.read_csv("raw-test-data.csv")

    sequences = pd.unique(df["sequence"])

    train_set_size = int(0.8 * len(sequences))
    train_sequences = sequences[0:train_set_size]
    train_df = pd.DataFrame(
        {
            "record_id": [n for n in range(0, train_set_size)],
            "sequence": train_sequences,
        }
    )

    val_set_size = int(0.1 * len(sequences))
    val_sequences = sequences[train_set_size : (train_set_size + val_set_size)]
    val_df = pd.DataFrame(
        {
            "record_id": [
                n for n in range(train_set_size, (train_set_size + val_set_size))
            ],
            "sequence": val_sequences,
        }
    )

    holdout_set_size = int(0.1 * len(sequences))
    holdout_sequences = sequences[
        (train_set_size + val_set_size) : (
            train_set_size + val_set_size + holdout_set_size
        )
    ]
    holdout_df = pd.DataFrame(
        {
            "record_id": [
                n
                for n in range(
                    (train_set_size + val_set_size),
                    (train_set_size + val_set_size + holdout_set_size),
                )
            ],
            "sequence": holdout_sequences,
        }
    )

    assert all(x not in holdout_df["sequence"] for x in train_df["sequence"])
    assert all(x not in val_df["sequence"] for x in train_df["sequence"])
    assert all(x not in holdout_df["sequence"] for x in val_df["sequence"])

    train_df.to_csv(
        os.path.join(this_dir, "example-data/big-easy-train.csv"), index=False
    )
    val_df.to_csv(os.path.join(this_dir, "example-data/big-easy-val.csv"), index=False)
    holdout_df.to_csv(
        os.path.join(this_dir, "example-data/big-easy-holdout.csv"), index=False
    )

    # also produce the out-of-sample sets
    out_of_sample_regex_trivial_sampler(
        2000, out_path=os.path.join(this_dir, "..", "trivial-out-of-sample.csv")
    )
    out_of_sample_regex_easier_sampler(
        2000, out_path=os.path.join(this_dir, "..", "easy-out-of-sample.csv")
    )
    out_of_sample_regex_harder_sampler(
        2000, out_path=os.path.join(this_dir, "..", "harder-out-of-sample.csv")
    )
