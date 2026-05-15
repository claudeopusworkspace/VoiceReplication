"""Run RVC's extract_feature_print.py with torch.load monkey-patched.

torch 2.6+ defaults `weights_only=True`, which rejects the fairseq HuBERT
checkpoint (contains pickled fairseq.data.dictionary.Dictionary). Same gotcha
the RVC inference adapter handles. We patch torch.load to default
`weights_only=False` before fairseq is imported.

All env vars and CLI args from the caller pass through to the wrapped script.
"""
import runpy
import sys
from pathlib import Path

import torch

_orig_torch_load = torch.load


def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_unsafe

repo_root = Path("/workspace/VoiceReplication/specialized/rvc")
sys.path.insert(0, str(repo_root))

# extract_feature_print.py reads sys.argv directly; we pass through everything
# after our own script path verbatim, since the bash caller invokes us with
# the same arg signature.
runpy.run_path(
    str(repo_root / "infer/modules/train/extract_feature_print.py"),
    run_name="__main__",
)
