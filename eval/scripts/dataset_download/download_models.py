import os
import yaml
from transformers import AutoModel, AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_models_from_config(config_dir):
    """
    Downloads all datasets specified in the YAML configuration files within a directory.

    Args:
        config_dir (str): The path to the directory containing the data config YAML files.
    """
    if not os.path.isdir(config_dir):
        logging.error(f"Configuration directory not found: {config_dir}")
        return

    logging.info(f"Scanning for model configurations in: {config_dir}")

    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            config_path = os.path.join(config_dir, filename)
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # Look for model path in common config structures
                model_path = None
                if "model_name" in config:
                    model_path = config["model_name"]
                    token = config["hf_kwargs"].get("token", None) if "hf_kwargs" in config else None

                if model_path:
                    logging.info(f"Found model path '{model_path}' in '{filename}'. Downloading...")
                    try:
                        # Download the model
                        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, token=token)
                        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=token)
                        logging.info(f"Successfully downloaded '{model_path}'.")
                    except Exception as e:
                        logging.error(f"Failed to download model '{model_path}'. Reason: {e}")
                else:
                    logging.warning(f"Could not find a model path in '{filename}'.")

            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file '{filename}': {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred with file '{filename}': {e}")


if __name__ == "__main__":
    # The script assumes it's run from the root of the AMPLIFY-evaluation directory.
    # The path to the config directory is relative to the root.
    model_config_directory = "conf/model"
    download_models_from_config(model_config_directory)
    logging.info("All models download attempts are complete.")
