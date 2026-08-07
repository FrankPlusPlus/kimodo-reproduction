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
    PYTHONUNBUFFERED=1 \
    KIMODO_STORAGE_ROOT=/mnt/kimodo

# Where your code will live inside the container
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
      cmake build-essential \
      gosu \
      zstd \
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
# image. Mount the PVC at KIMODO_STORAGE_ROOT (default: /mnt/kimodo), or set
# that environment variable to the platform-specific mount path at runtime.
COPY configs /workspace/configs
COPY resources /workspace/resources
COPY scripts /workspace/scripts
COPY benchmark /workspace/benchmark
RUN find /workspace/scripts -type f -name '*.sh' -exec chmod +x {} +

# Use the docker-entrypoint script, to allow the docker to run as the actual user instead of root
COPY kimodo/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

# Keep the image hardware-neutral by default.  The dispatcher stays alive in
# idle mode, while Kubernetes can either override the command with a concrete
# launcher or select an explicit KIMODO_CONTAINER_MODE.  Importantly, merely
# creating a Pod must not require 16 H200s, RDMA, or a mounted production PVC.
ENTRYPOINT ["docker-entrypoint"]
CMD ["/workspace/scripts/container_start.sh"]
