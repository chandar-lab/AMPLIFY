import os
import hydra
from omegaconf import DictConfig
from sm_training import trainer


@hydra.main(version_base=None, config_path="/opt/ml/code/conf", config_name="config")
def pipeline(cfg: DictConfig):
    # Start training
    trainer(cfg)


if __name__ == "__main__":
    pipeline()
