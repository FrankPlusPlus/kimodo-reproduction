# syntax=docker/dockerfile:1.7

# Keep the known project base as the safe default. Company CI may override it
# with an approved, digest-pinned NGC image after checking the host driver.
ARG KIMODO_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${KIMODO_PYTORCH_IMAGE}

ARG KIMODO_GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=${KIMODO_GIT_COMMIT}
ENV KIMODO_IMAGE_GIT_COMMIT=${KIMODO_GIT_COMMIT}

# Avoid some interactive prompts + make pip quieter/reproducible-ish
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Where your code will live inside the container
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      cmake build-essential \
      gosu \
      libibverbs1 ibverbs-providers rdma-core \
      infiniband-diags perftest ibutils ibverbs-utils \
    && rm -rf /var/lib/apt/lists/*

# Some base images ship a broken `/usr/local/bin/cmake` shim (from a partial pip install),
# which shadows `/usr/bin/cmake` and breaks builds that invoke `cmake` (e.g. MotionCorrection).
# Prefer the system cmake.
RUN rm -f /usr/local/bin/cmake || true

# Install from docker_requirements.txt: kimodo editable (-e .),
# but MotionCorrection non-editable (./MotionCorrection). The -e . line ensures [project.scripts]
# from pyproject.toml are installed (kimodo_gen, kimodo_demo, kimodo_textencoder).
# SKIP_MOTION_CORRECTION_IN_SETUP=1 so setup.py does not bundle motion_correction; it is
# installed separately from ./MotionCorrection in the requirements file (non-editable).
COPY docker_requirements.txt /workspace/docker_requirements.txt
COPY setup.py /workspace/setup.py
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY kimodo /workspace/kimodo
COPY MotionCorrection /workspace/MotionCorrection

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && SKIP_MOTION_CORRECTION_IN_SETUP=1 python -m pip install -r docker_requirements.txt \
 && python -m pip check \
 && python -c "import kimodo, torch; print('kimodo image smoke:', torch.__version__, torch.version.cuda)"

# Training method/config files and operational launchers are runtime inputs.
# Datasets, prepared caches, and checkpoints deliberately stay outside the
# image and are mounted by the scheduler at /mnt/kimodo (or another path
# selected through KIMODO_PATHS_CONFIG).
COPY configs /workspace/configs
COPY resources /workspace/resources
COPY scripts /workspace/scripts
COPY benchmark /workspace/benchmark
RUN chmod +x /workspace/scripts/*.sh /workspace/scripts/resources/*.sh

# Use the docker-entrypoint script, to allow the docker to run as the actual user instead of root
COPY kimodo/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

# The default command is the company launcher; its training config and paths
# remain runtime-selectable, so the same image supports V1, V2 and eval Pods.
ENTRYPOINT ["docker-entrypoint"]
CMD ["/workspace/scripts/train_company_16h200.sh"]
