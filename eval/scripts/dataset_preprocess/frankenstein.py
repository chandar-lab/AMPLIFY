import os
import re
import random
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_path", default=None, type=str, required=False)
    parser.add_argument("--vocabulary_file", type=str, required=True)
    parser.add_argument("--min_len", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=512)

    args = parser.parse_args()

    input_name = os.path.basename(args.input_file).replace(".txt", "")

    # Load vocabulary
    with open(args.vocabulary_file, "rt") as f:
        vocabulary = [line.strip() for line in f]
    valid_chars = set(vocabulary)

    # Build filtered and cleaned text
    cleaned_text = []
    with open(args.input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            # Remove non-alpha characters, uppercase, and filter by vocabulary
            filtered = "".join(c for c in line.upper() if c.isalpha() and c in valid_chars)
            if filtered:
                cleaned_text.append(filtered)

    # Join all text and remove implicit newlines
    full_text = "".join(cleaned_text)

    # Split into sequences
    sequences = []
    i = 0
    while i < len(full_text):
        line_length = random.randint(args.min_len, args.max_len - 1)
        chunk = full_text[i : i + line_length]
        sequences.append("M" + chunk)  # Add methionine
        i += line_length

    if args.output_path is None:
        csv_file = args.input_file.replace(".txt", ".csv")
    else:
        csv_file = os.path.join(args.output_path, input_name + ".csv")
    with open(csv_file, "w") as f:
        f.write("name,sequence\n")
        for idx, seq in enumerate(sequences, start=1):
            f.write(f"|{input_name}_{idx:05d},{seq}\n")
