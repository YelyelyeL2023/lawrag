# lawrag application image.
#
# Two targets, because the weights stack is ~25x larger than everything else and
# most of the pipeline does not need it:
#
#   runtime            — CLI, API, console, ingest, retrieval, llama.cpp client.
#                        ~390 MB. Enough for a machine whose config uses the
#                        placeholder embedder or a remote embedding service.
#   models   (default) — runtime plus torch + FlagEmbedding, for bge-m3 and the
#                        bge reranker. ~10 GB; needs the NVIDIA container
#                        toolkit on the host to reach a GPU. Last stage, so a
#                        plain `docker build` with no --target produces this.
#
# Air gap: build on the connected side, `docker save` the images, carry the
# tarball, `docker load` on the target. No wheelhouse to assemble, no pip on
# the target machine, no virtualenv anywhere.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp is what torch and FlagEmbedding link against; it is in the base layer
# so that the `runtime` and `models` targets share it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first: source edits must not invalidate the wheel install.
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts

# Installed into the image's own site-packages. There is no virtualenv: the
# container *is* the isolation boundary, and a venv inside it would only add a
# second interpreter for a missing module to hide behind.
#
# Editable so that bind-mounting ./src over /app/src during development runs the
# mounted code; without the mount the copy baked in above is what runs.
RUN pip install -e ".[dev]"

# Data and config are bind-mounted in compose; create them so the image also
# works standalone.
RUN mkdir -p /app/data /app/config /models

# Non-root. The uid is fixed so a bind-mounted ./data keeps stable ownership
# across rebuilds on Linux hosts.
RUN useradd --create-home --uid 10001 lawrag \
 && chown -R lawrag:lawrag /app /models
USER lawrag

ENV LAWRAG_CONFIG=/app/config/default.yaml
# Read back by the console's /api/stats, so an operator looking at a missing
# module can tell whether the process answering them is the one compose manages.
ENV LAWRAG_IN_CONTAINER=1

# No curl in the image: one fewer package, and the check runs on the same
# interpreter the server does.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["lawrag", "serve", "--host", "0.0.0.0", "--port", "8000"]


# --------------------------------------------------------------- runtime --
FROM base AS runtime


# ---------------------------------------------------------------- models --
# torch's default PyPI wheel for Linux is the CUDA build. For a CPU-only host,
# build with:
#   --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
FROM base AS models

# A build arg, not an ENV: pip reads it from the build environment here and it
# does not linger in the shipped image.
ARG PIP_EXTRA_INDEX_URL=""

USER root
RUN pip install -e ".[models]" && chown -R lawrag:lawrag /app
USER lawrag

# HF_HOME lands on the models volume rather than in the container's writable
# layer, so nothing re-downloads after a `docker compose down`. On the
# air-gapped side nothing downloads at all: model_path points at a local dir.
ENV HF_HOME=/models/.hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
