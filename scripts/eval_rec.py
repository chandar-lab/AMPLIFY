import hydra
from omegaconf import DictConfig

from amplify import rec_eval_trainer


@hydra.main(version_base=None, config_path="../conf/amplify", config_name="config_eval_rec")
def pipeline(cfg: DictConfig) -> None:
    """Hydra entrypoint that dispatches to ``amplify.rec_eval_trainer``."""
    rec_eval_trainer(cfg)


if __name__ == "__main__":
    pipeline()
