from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .metric import TrainingLogger
from .model import (
    AMPLIFYConfig,
    AMPLIFYForMaskedLM,
    AMPLIFYForSequenceClassification,
    AMPLIFYForTokenClassification,
    AMPLIFYModel,
    ProteinTokenizer,
    RecAMPLIFY,
    RecAMPLIFYConfig,
    TokenizerConfig,
)
from .trainer import rec_eval_trainer, rec_trainer, trainer

# Register once at import time so Hub checkpoints work with Auto* classes.
AutoConfig.register("amplify", AMPLIFYConfig)
AutoModel.register(AMPLIFYConfig, AMPLIFYModel)
AutoModelForMaskedLM.register(AMPLIFYConfig, AMPLIFYForMaskedLM)
AutoModelForSequenceClassification.register(AMPLIFYConfig, AMPLIFYForSequenceClassification)
AutoModelForTokenClassification.register(AMPLIFYConfig, AMPLIFYForTokenClassification)
AutoTokenizer.register(AMPLIFYConfig, fast_tokenizer_class=ProteinTokenizer)

# Pre-release checkpoints (model_type "amplify_srm") load with RecAMPLIFY.from_pretrained.
AutoConfig.register("recamplify", RecAMPLIFYConfig)
AutoModelForMaskedLM.register(RecAMPLIFYConfig, RecAMPLIFY)
AutoTokenizer.register(RecAMPLIFYConfig, fast_tokenizer_class=ProteinTokenizer)

__all__ = [
    "AMPLIFYConfig",
    "AMPLIFYForMaskedLM",
    "AMPLIFYForSequenceClassification",
    "AMPLIFYForTokenClassification",
    "AMPLIFYModel",
    "ProteinTokenizer",
    "RecAMPLIFY",
    "RecAMPLIFYConfig",
    "TokenizerConfig",
    "TrainingLogger",
    "rec_eval_trainer",
    "rec_trainer",
    "trainer",
]
