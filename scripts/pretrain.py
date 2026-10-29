import hydra
from omegaconf import DictConfig

from amplify import trainer


@hydra.main(version_base=None, config_path="../conf/amplify", config_name="config")
def pipeline(cfg: DictConfig) -> None:
    """Hydra entrypoint that dispatches to ``amplify.trainer``."""
    # Hydra composes config defaults + CLI overrides into ``cfg``.
    # Trainer consumes that config and runs the full train/eval loop.
    trainer(cfg)


if __name__ == "__main__":
    pipeline()
