"""lawrag — retrieval-augmented lookup over Kazakhstan's Constitution and Codes.

Designed for a fully air-gapped deployment: every model is local, the corpus
arrives as signed offline snapshot bundles, and nothing in the query path makes
an outbound network call.
"""

import os

# transformers treats a model_path that does not exist as a Hub repo id and
# tries to download it. On the air-gapped side that must fail fast, not hang on
# the network. Set before FlagEmbedding / transformers are imported (they read
# these at import time); an explicit export still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.1.0"
