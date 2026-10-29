"""Single import seam between the eval suite and the training package.

The eval suite loads exactly one model class, `RecAMPLIFY`, from the sibling `src/amplify`
package.  Everything that needs the model imports it from here, so the coupling to the
training repo is one line in one file rather than `sys.path` walking scattered across five
scripts (which is what the original did, with three different parent depths).

`amplify` must be importable: either `pip install -e .` at the repo root, or
`PYTHONPATH=<repo>/src`, which `jobs/run_eval.sh` sets.
"""

from amplify.model import RecAMPLIFY, RecAMPLIFYConfig
from amplify.model.tokenizer import ProteinTokenizer, TokenizerConfig

__all__ = ["RecAMPLIFY", "RecAMPLIFYConfig", "ProteinTokenizer", "TokenizerConfig"]
