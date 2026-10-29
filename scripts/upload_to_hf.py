import argparse
import json
import os

from huggingface_hub import HfApi


_AMPLIFY_AUTO_MAP = {
    "AutoConfig": "amplify.AMPLIFYConfig",
    "AutoModel": "amplify.AMPLIFYModel",
    "AutoModelForMaskedLM": "amplify.AMPLIFYForMaskedLM",
    "AutoModelForSequenceClassification": "amplify.AMPLIFYForSequenceClassification",
    "AutoModelForTokenClassification": "amplify.AMPLIFYForTokenClassification",
}
# Written by the recAMPLIFY trainer and by scripts/export_hf.py; recamplify.py imports amplify.py.
_RECAMPLIFY_AUTO_MAP = {
    "AutoConfig": "recamplify.RecAMPLIFYConfig",
    "AutoModelForMaskedLM": "recamplify.RecAMPLIFY",
}


def _is_self_contained_checkpoint(checkpoint_dir: str) -> bool:
    """Return ``True`` if *checkpoint_dir* is already portable for Auto* loading."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    tokenizer_config_path = os.path.join(checkpoint_dir, "tokenizer_config.json")
    if not (os.path.isfile(config_path) and os.path.isfile(tokenizer_config_path)):
        return False

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        with open(tokenizer_config_path, encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    auto_map = config.get("auto_map", {})
    if config.get("model_type") == "recamplify":
        expected, code_files = _RECAMPLIFY_AUTO_MAP, ("amplify.py", "recamplify.py", "tokenizer.py")
    else:
        expected, code_files = _AMPLIFY_AUTO_MAP, ("amplify.py", "tokenizer.py")
    if any(auto_map.get(key) != value for key, value in expected.items()):
        return False

    tokenizer_auto_map = tokenizer_config.get("auto_map", {})
    return "AutoTokenizer" in tokenizer_auto_map and all(
        os.path.isfile(os.path.join(checkpoint_dir, name)) for name in code_files
    )


def main() -> None:
    """Upload a self-contained AMPLIFY or recAMPLIFY checkpoint directory to a Hub model repo."""
    # Parse CLI options and environment-based defaults.
    parser = argparse.ArgumentParser(description="Upload an AMPLIFY or recAMPLIFY checkpoint to the Hugging Face Hub.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the HF checkpoint directory.")
    parser.add_argument("--repo_id", type=str, required=True, help="HuggingFace repo ID (e.g. org/model-name).")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", action="store_true", help="Create/update a private repo.")
    visibility.add_argument("--public", action="store_true", help="Create/update a public repo.")
    parser.add_argument(
        "--revision",
        type=str,
        default=os.getenv("HF_HUB_REVISION"),
        help="Target branch/tag/commit on the Hub (default: HF_HUB_REVISION env var).",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload AMPLIFY checkpoint",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token (default: HF_TOKEN or HUGGINGFACE_HUB_TOKEN env var).",
    )
    parser.set_defaults(private=None)
    args = parser.parse_args()
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)

    # Ensure the target repo exists with the requested visibility.
    private = True if args.private else False if args.public else None
    api = HfApi(token=args.token)
    create_repo_kwargs = {"repo_id": args.repo_id, "repo_type": "model", "exist_ok": True}
    if private is not None:
        create_repo_kwargs["private"] = private
    api.create_repo(**create_repo_kwargs)

    if not _is_self_contained_checkpoint(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} is not a self-contained HF checkpoint. "
            "Export it with scripts/export_hf.py (recAMPLIFY) or re-save it with the current "
            "pretraining checkpoint saver before uploading."
        )

    print(f"Uploading self-contained checkpoint from {checkpoint_dir}...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=args.token,
        commit_message=args.commit_message,
    )

    # Print a direct link to the uploaded model card/repo.
    print(f"Done! Model available at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
