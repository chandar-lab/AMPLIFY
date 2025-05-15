import csv
import json
import os
import pickle

import numpy as np
import torch

from amplify.inference.embeddings import (
    cosine_similarities,
    Embedder,
)

from .fixtures import load_model, temporary_file_path

this_dir = os.path.dirname(__file__)

reference_model_checkpoint_path = os.path.join(
    this_dir, "reference-model/saved/model.safetensors"
)
reference_model_config_path = os.path.join(this_dir, "reference-model/config.yaml")

reference_model, reference_tokenizer = load_model(
    reference_model_checkpoint_path, reference_model_config_path
)


def test__embedder__creates_reproducible_embeddings():
    """
    These are determined from an initial baseline,
    created by running the code in `tests/produce-reference-model.py`
    and using the code that originally produced the AMPLIFY figures.

    This model is for testing the inference API only,
    and does _not_ reproduce the AMPLIFY results
    """
    datapoint_count = 10
    
    model, tokenizer = (reference_model.eval(), reference_tokenizer)

    sequence_file_path = os.path.join(this_dir, "example-data/easy-task-val.csv")
    with open(sequence_file_path, "r") as data_source:
        reader = csv.reader(data_source)

        sequences = list()
        for _ in range(datapoint_count):
            sequences.append(next(reader)[1])

    embeddings = list()

    embedder = Embedder(model, tokenizer)
    for sequence in sequences:
        embedding = embedder.embed(sequence)
        embeddings.append(embedding.tolist())

    with open(
        os.path.join(this_dir, "example-data/reference-embeddings.json"), "w"
    ) as f:
        json.dump(embeddings, f)

    with open(
        os.path.join(this_dir, "example-data/reference-embeddings.json"), "r"
    ) as f:
        expected_embeddings = json.load(f)

    assert len(expected_embeddings) == len(embeddings)

    for expected, actual in zip(expected_embeddings, embeddings):
        assert (torch.as_tensor(expected) - torch.as_tensor(actual)).abs().max() < 1e-9


def test__cosine_similarities__obeys_basic_identities_of_mathematical_definition():

    dtype = torch.float

    tol = 1e-12

    reference_vector = torch.tensor([1, 0, 0], dtype=dtype)
    comparison_vectors = [
        torch.tensor([0, 1, 0], dtype=dtype),
        torch.tensor([0, 1, 1], dtype=dtype),
        torch.tensor([0, 0, 1], dtype=dtype),
    ]

    similarities = cosine_similarities(reference_vector, comparison_vectors)
    assert all(x < tol for x in similarities)

    assert all(
        1.0 - x < tol for x in cosine_similarities(reference_vector, [reference_vector])
    )
    assert all(
        -1.0 - x < tol
        for x in cosine_similarities(reference_vector, [-reference_vector])
    )


def test__cosine_similarities__agrees_with_values_produced_by_a_naive_implementation():

    def _cosine_similarity(reference_vector, comparison_vector):

        dot_product = np.dot(comparison_vector, reference_vector)
        lnorm_A = np.linalg.norm(comparison_vector)
        lnorm_B = np.linalg.norm(reference_vector)

        return dot_product / (lnorm_A * lnorm_B)

    scale = 50
    rng = np.random.default_rng()
    input_vectors = scale * rng.uniform(-1, 0, (20, 100))
    reference_vector = input_vectors[0]
    comparison_vectors = input_vectors[1:]

    expected_values = np.array(
        [_cosine_similarity(reference_vector, x) for x in comparison_vectors]
    )

    actual_values = cosine_similarities(reference_vector, comparison_vectors)

    tol = 1e-12

    try:
        assert (np.absolute(expected_values - actual_values)).max() < tol
    except AssertionError:

        error_dump_path = os.path.abspath("test-error-dump.pkl")
        with open(error_dump_path, "wb") as f:
            pickle.dump(input_vectors, f)
            raise AssertionError(
                f"Expected values differed by more than {tol} on at least one dimension; original input dumped to {error_dump_path}"
            )


def test__dump__dumps_the_computed_embeddings_to_a_pickle(temporary_file_path):

    model, tokenizer = (reference_model.eval(), reference_tokenizer)
    embedder = Embedder(model, tokenizer)

    sequence_file_path = os.path.join(this_dir, "example-data/easy-task-val.csv")
    with open(sequence_file_path, "r") as data_source:
        reader = csv.reader(data_source)
        sequences = [row[1] for row in reader]

    expected_embeddings = [embedder.embed(x) for x in sequences]

    embedder.dump(sequences, out_path=temporary_file_path)

    assert os.path.exists(temporary_file_path)

    with open(temporary_file_path, "rb") as f:
        actual_embeddings = pickle.load(f)

    assert isinstance(actual_embeddings, list)
    assert len(expected_embeddings) == len(actual_embeddings)

    for expected, actual in zip(expected_embeddings, actual_embeddings):
        assert isinstance(actual, type(expected))
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        print(expected.shape)
        assert (
            np.max(np.abs(actual - expected)) < 1e-9
        ), f"{(actual - expected).abs().max()}"


def test__load__reproduces_the_results_of_self_dot_dump_given_a_filepath(
    temporary_file_path,
):

    model, tokenizer = (reference_model.eval(), reference_tokenizer)
    embedder = Embedder(model, tokenizer)

    sequence_file_path = os.path.join(this_dir, "example-data/easy-task-val.csv")
    with open(sequence_file_path, "r") as data_source:
        reader = csv.reader(data_source)
        sequences = [row[1] for row in reader]

    expected_embeddings = [embedder.embed(x) for x in sequences]

    embedder.dump(sequences, out_path=temporary_file_path)

    assert os.path.exists(temporary_file_path)

    actual_embeddings = Embedder.load(temporary_file_path)

    assert isinstance(actual_embeddings, list)
    assert len(expected_embeddings) == len(actual_embeddings)

    for expected, actual in zip(expected_embeddings, actual_embeddings):
        assert isinstance(actual, type(expected))
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        assert (
            np.max(np.abs(actual - expected)) < 1e-9
        ), f"{(actual - expected).abs().max()}"


def test__load__reproduces_the_results_of_self_dot_dump_given_an_io_object(
    temporary_file_path,
):

    model, tokenizer = (reference_model.eval(), reference_tokenizer)
    embedder = Embedder(model, tokenizer)

    sequence_file_path = os.path.join(this_dir, "example-data/easy-task-val.csv")
    with open(sequence_file_path, "r") as data_source:
        reader = csv.reader(data_source)
        sequences = [row[1] for row in reader]

    expected_embeddings = [embedder.embed(x) for x in sequences]

    embedder.dump(sequences, out_path=temporary_file_path)

    assert os.path.exists(temporary_file_path)

    with open(temporary_file_path, "rb") as f:
        actual_embeddings = Embedder.load(f)

    assert isinstance(actual_embeddings, list)
    assert len(expected_embeddings) == len(actual_embeddings)

    for expected, actual in zip(expected_embeddings, actual_embeddings):
        assert isinstance(actual, type(expected))
        assert expected.dtype == actual.dtype
        assert expected.shape == actual.shape
        assert (
            np.max(np.abs(actual - expected)) < 1e-9
        ), f"{(actual - expected).abs().max()}"
