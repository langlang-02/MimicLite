# Build from repository root: docker build -f local/mjlab/Dockerfile -t IMAGE .
ARG BASE_IMAGE=vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:2.10.0-cu12.9.1-py3.11-ubuntu22.04
FROM ${BASE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv==0.12.1
WORKDIR /opt/mimiclite
COPY active-adaptation ./active-adaptation
COPY mimic-lite ./mimic-lite
COPY local/mjlab ./local/mjlab
ENV MIMICLITE_ROOT=/opt/mimiclite
RUN bash -c 'source local/mjlab/env.sh && uv venv --python 3.12.13 local/mjlab/.venv && uv pip sync --python local/mjlab/.venv/bin/python local/mjlab/requirements.lock && uv pip check --python local/mjlab/.venv/bin/python && HF_HUB_OFFLINE=0 python local/mjlab/fetch_assets.py'
# The platform Command invokes entrypoint.sh explicitly; default never trains.
CMD ["bash", "-lc", "source local/mjlab/env.sh && python -c 'import torch, mjlab; print(torch.__version__)'"]
