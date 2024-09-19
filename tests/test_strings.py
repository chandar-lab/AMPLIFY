from amplify.inference.strings import (
    aa_sequences_from_text,
    filter_non_amino_acid_chars,
)

from .fixtures import temporary_file_path


def test__filter_amino_acids__contains_only_uppercase_amino_acid_characters():
    text_string = "Sink all coffins and all hearses to one common pool!"

    expected = "SINKALLCFFINSANDALLHEARSESTNECMMNPL"

    assert expected == filter_non_amino_acid_chars(text_string)


def test__aa_sequences_from_text__returns_token_split_sequences_of_amino_acids(
    temporary_file_path,
):
    example_text = """
    The bat sat on the hat.
    The hat sat on the cat.
    The cat sat on the mat.
    And that was that.
    """

    with open(temporary_file_path, "w") as f:
        f.write(example_text)

    expected_output = [
        "THEATSATNTHEHAT",
        "THEHATSATNTHECAT",
        "THECATSATNTHEMAT",
        "ANDTHATWASTHAT",
    ]

    assert expected_output == list(
        aa_sequences_from_text(temporary_file_path, upper_cut=20, lower_cut=5)
    )


def test__aa_sequences_from_text__drops_sequences_above_a_given_length(
    temporary_file_path,
):
    example_text = """
    The bat sat on the hat.
    The hat sat on the cat.
    Honestly, this is just ridiculous, and a lot of extra typing besides.
    The cat sat on the mat.
    And that was that.
    """

    with open(temporary_file_path, "w") as f:
        f.write(example_text)

    expected_output = [
        "THEATSATNTHEHAT",
        "THEHATSATNTHECAT",
        "THECATSATNTHEMAT",
        "ANDTHATWASTHAT",
    ]

    assert expected_output == list(
        aa_sequences_from_text(temporary_file_path, upper_cut=20, lower_cut=5)
    )


def test__aa_sequences_from_text__concatenates_sequences_below_a_given_length(
    temporary_file_path,
):
    example_text = """
    The bat sat on the hat.
    The hat sat on the cat.
    Honestly, this is just ridiculous, and a lot of extra typing besides.
    The.
    cat.
    sat.
    on.
    the.
    mat.
    And that was that.
    ok.
    wow.
    neat.
    fantastic.
    this part is extra
    """

    with open(temporary_file_path, "w") as f:
        f.write(example_text)

    expected_output = [
        "THEATSATNTHEHAT",
        "THEHATSATNTHECAT",
        "THECATSATNTHEMAT",
        "ANDTHATWASTHAT",
        "KWWNEATFANTASTIC",
        "THISPARTISETRA",
    ]

    # the function should produce a generator;
    # create a new generator for each of the assert calls
    def _get_result():
        return aa_sequences_from_text(temporary_file_path, upper_cut=20, lower_cut=13)

    # materialized lists should be equal
    aa_sequence = _get_result()
    assert expected_output == list(aa_sequence)

    # iterating over the object should be possible
    aa_sequence = _get_result()
    for expected in expected_output:
        assert expected == next(aa_sequence)


def test__aa_sequences_from_text__reads_in_very_small_chunks_with_correct_sentence_output(
    temporary_file_path,
):
    example_text = """
    The bat sat on the hat.
    The hat sat on the cat.
    Honestly, this is just ridiculous, and a lot of extra typing besides.
    The.
    cat.
    sat.
    on.
    the.
    mat.
    And that was that.
    ok.
    wow.
    neat.
    fantastic.
    this part is extra
    """

    with open(temporary_file_path, "w") as f:
        f.write(example_text)

    expected_output = [
        "THEATSATNTHEHAT",
        "THEHATSATNTHECAT",
        "THECATSATNTHEMAT",
        "ANDTHATWASTHAT",
        "KWWNEATFANTASTIC",
        "THISPARTISETRA",
    ]

    # the function should produce a generator;
    # create a new generator for each of the assert calls
    def _get_result():
        return aa_sequences_from_text(
            temporary_file_path, upper_cut=20, lower_cut=13, io_chunk_size=2
        )

    # materialized lists should be equal
    aa_sequence = _get_result()
    assert expected_output == list(aa_sequence)

    # iterating over the object should be possible
    aa_sequence = _get_result()
    for expected in expected_output:
        assert expected == next(aa_sequence)


def test__aa_sequences_from_text__reads_in_very_large_chunks_with_correct_sentence_output(
    temporary_file_path,
):
    example_text = """
    The bat sat on the hat.
    The hat sat on the cat.
    Honestly, this is just ridiculous, and a lot of extra typing besides.
    The.
    cat.
    sat.
    on.
    the.
    mat.
    And that was that.
    ok.
    wow.
    neat.
    fantastic.
    this part is extra
    """

    with open(temporary_file_path, "w") as f:
        f.write(example_text)

    expected_output = [
        "THEATSATNTHEHAT",
        "THEHATSATNTHECAT",
        "THECATSATNTHEMAT",
        "ANDTHATWASTHAT",
        "KWWNEATFANTASTIC",
        "THISPARTISETRA",
    ]

    # the function should produce a generator;
    # create a new generator for each of the assert calls
    def _get_result():
        return aa_sequences_from_text(
            temporary_file_path, upper_cut=20, lower_cut=13, io_chunk_size=8096
        )

    # materialized lists should be equal
    aa_sequence = _get_result()
    assert expected_output == list(aa_sequence)

    # iterating over the object should be possible
    aa_sequence = _get_result()
    for expected in expected_output:
        assert expected == next(aa_sequence)
