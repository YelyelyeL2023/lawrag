"""lawrag — retrieval-augmented lookup over Kazakhstan's Constitution and Codes.

Designed for a fully air-gapped deployment: every model is local, the corpus
arrives as signed offline snapshot bundles, and nothing in the query path makes
an outbound network call.
"""

__version__ = "0.1.0"
