"""One lock for every local model forward pass.

The API serves requests from a thread pool. On Apple's MPS backend two threads
running the embedder and the reranker at once abort the whole process inside
Metal ("failed assertion _status < MTLCommandBufferStatusCommitted"): a double
click in the console was enough. Serialising forward passes costs nothing on a
single-GPU workstation, where they would contend for the device anyway; the
LLM call, which is network-bound, stays outside the lock.
"""

from __future__ import annotations

import threading

MODEL_LOCK = threading.Lock()
