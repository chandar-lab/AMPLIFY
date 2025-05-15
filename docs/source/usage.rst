Usage
=====

Inference
---------

The trained model can be wrapped in an instance of `amplify.inference.Predictor`
for direct access to the underlying inference API, as in this example:

.. code-block:: python

    import amplify

    # Load the model
    config_path = "/local/path/to/model/config/config.yaml"
    checkpoint_file = "/local/path/to/model/checkpoint"

    model, tokenizer = amplify.AMPLIFY.load(checkpoint_file, config_path)

    # Link the model to the inference API:
    predictor = amplify.inference.Predictor(model, tokenizer, device=device)

    # Calculate logits for a sequence
    sequence = "MSVVGIDLGFQSCYVAVARAGGIETIANEYSDRCTPACISFGPKNR"
    logits = predictor.logits(sequence)

    # Get the embedding for a given sequence:
    embedder = (model, tokenizer)
    sequence_embedding = predictor.embed(sequence)

    # Compare the sequence to several other sequences
    other_sequences = [
        "AACGGEVWVTDEAAAAA",
        "AAAAACGGGVWWTDEAAAAA",
        "AAAADGGVWVTECDA",
    ]

    other_sequence_embeddings = [predictor.embed(x) for x in other_sequences]

    import numpy as np
    other_embedding_mean = np.mean(np.concatenate(other_sequence_embeddings), axis=0)
    similarities = amplify.inference.cosine_similarities(
        other_embedding_mean,
        sequence_embedding,
    )

Measuring Similarity to Human Language Text
-------------------------------------------

The package includes a public-facing function `compare_sequences_to_human_text` that reproduces cosine similarities such as those in the "Frankenstein" analysis in the AMPLIFY paper. Given a version of the model and a text
file, it can produce similarity measures between a set of sequences and the text-embedding-average, as in the example below:

.. code-block:: python

    import amplify

    # load the model

    config_path = "/local/path/to/model/config/config.yaml"
    checkpoint_file = "/local/path/to/model/checkpoint/model.safetensors"

    model, tokenizer = amplify.AMPLIFY.load(checkpoint_file, config_path)
    model = model.eval()

    example_target_sequences = [
        "AACGGEVWVTDEAAAAA",
        "AAAAACGGGVWWTDEAAAAA",
        "AAAADGGVWVTECDA",
    ]

    # calculate the similarities
    text_path = "/local/path/to/text_source/example.txt"
    similarity_measures = amplify.inference.compare_sequences_to_human_text(
        tokenizer=tokenizer,
        model=model,
        text_path=text_path,
        target_sequences=example_target_sequences,
    )


Training
-----------

Model configuration is managed by Hydra, with `amplify.trainer.trainer`
as the top-level entry point for training.

Given a well-formed Hydra configuration, a minimal script
for initiating a training run could look like this:

.. code-block:: python

    import hydra
    from omegaconf import DictConfig

    from src.amplify import trainer


    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def pipeline(cfg: DictConfig):
        trainer(cfg)


    if __name__ == "__main__":
        pipeline()