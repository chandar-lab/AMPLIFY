import os
import yaml
from datasets import load_dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_datasets_from_config(config_dir):
    """
    Downloads all datasets specified in the YAML configuration files within a directory.

    Args:
        config_dir (str): The path to the directory containing the data config YAML files.
    """
    if not os.path.isdir(config_dir):
        logging.error(f"Configuration directory not found: {config_dir}")
        return

    logging.info(f"Scanning for dataset configurations in: {config_dir}")

    for filename in os.listdir(config_dir):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            config_path = os.path.join(config_dir, filename)
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

                # Look for dataset path in common config structures
                dataset_path = None
                if "data" in config and "dataset_name" in config["data"]:
                    dataset_path = config["data"]["dataset_name"]
                elif "dataset_name" in config:
                    dataset_path = config["dataset_name"]
                elif "path" in config:
                    dataset_path = config["path"]

                if dataset_path:
                    logging.info(f"Found dataset path '{dataset_path}' in '{filename}'. Downloading...")
                    try:
                        # Download the dataset
                        load_dataset(dataset_path, trust_remote_code=True)
                        logging.info(f"Successfully downloaded '{dataset_path}'.")
                    except Exception as e:
                        logging.error(f"Failed to download dataset '{dataset_path}'. Reason: {e}")
                else:
                    logging.warning(f"Could not find a dataset path in '{filename}'.")

            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file '{filename}': {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred with file '{filename}': {e}")


if __name__ == "__main__":
    # The script assumes it's run from the root of the AMPLIFY-evaluation directory.
    # The path to the config directory is relative to the root.
    data_config_directory = "conf/data_config"
    download_datasets_from_config(data_config_directory)
    logging.info("All dataset download attempts are complete.")
