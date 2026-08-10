# syntax=docker/dockerfile:1.7
#
# Company (Hanhai H200) training image.
# Highest-confidence multi-node stack aligns with the cluster-proven CUDA 13
# + R575 host pattern: CUDA devel base, compat libraries first on
# LD_LIBRARY_PATH, RDMA verbs/rdmacm/numa, PyTorch cu130. Kimodo app code
# still prefers PVC at runtime (KIMODO_CODE_ROOT).
#
# Optional legacy path: set KIMODO_PYTORCH_IMAGE to an NGC pytorch tag and
# KIMODO_REUSE_BASE_ENV=1 for source-only rebuilds on that base.

ARG KIMODO_PYTORCH_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
# Pin amd64: Apple Silicon defaults to arm64 and that image cannot run on H200.
FROM --platform=linux/amd64 ${KIMODO_PYTORCH_IMAGE}

ARG KIMODO_REUSE_BASE_ENV=0
ARG KIMODO_PYTHON_VERSION=3.12
ARG KIMODO_TORCH_VERSION=2.11.0
ARG KIMODO_TORCHVISION_VERSION=0.26.0
ARG KIMODO_TORCHAUDIO_VERSION=2.11.0
ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_BREAK_SYSTEM_PACKAGES=1

# CUDA runtime/compiler paths. compat must precede host-mounted driver libs
# when the image CUDA toolkit is newer than what older docs assume; H200
# nodes in this company fleet run R575-class drivers with CUDA 13 images.
# Match cluster-proven CUDA13 images: compat before host-mounted driver libs.
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TZ=Asia/Shanghai \
    KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction \
    KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction \
    NCCL_IB_HCA=mlx5 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    NCCL_ASYNC_ERROR_HANDLING=1

WORKDIR /workspace

# Prefer a China-reachable Ubuntu mirror; archive.ubuntu.com often 502 from here.
RUN set -eux; \
    for f in /etc/apt/sources.list /etc/apt/sources.list.d/ubuntu.sources; do \
      [ -f "$f" ] || continue; \
      sed -i \
        -e 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
        -e 's|https://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
        -e 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
        -e 's|https://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
        "$f"; \
    done; \
    printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-kimodo-retries

# System deps + RDMA (runtime + headers, matching cluster-proven images).
RUN if [ "${KIMODO_REUSE_BASE_ENV}" = 1 ]; then \
      command -v git && command -v curl && command -v cmake && command -v gosu \
      && command -v ibv_devices \
      && if ! command -v zstd >/dev/null 2>&1 || ! command -v sshd >/dev/null 2>&1; then \
           apt-get update \
           && apt-get install -y --no-install-recommends zstd openssh-server \
           && rm -rf /var/lib/apt/lists/*; \
         fi; \
    else \
      apt-get update \
      && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates \
        cmake build-essential \
        software-properties-common \
        gosu \
        zstd openssh-server \
        vim \
        tzdata \
        iproute2 \
        python${KIMODO_PYTHON_VERSION} \
        python${KIMODO_PYTHON_VERSION}-dev \
        libibverbs1 \
        libibverbs-dev \
        ibverbs-providers \
        rdma-core \
        librdmacm1 \
        librdmacm-dev \
        libnuma1 \
        libnuma-dev \
        numactl \
        infiniband-diags \
        ibverbs-utils \
        perftest \
        ibutils \
      && (apt-get install -y --no-install-recommends libmlx5-1 || true) \
      && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
      && echo "Asia/Shanghai" > /etc/timezone \
      && ln -sf /usr/bin/python${KIMODO_PYTHON_VERSION} /usr/bin/python3 \
      && ln -sf /usr/bin/python${KIMODO_PYTHON_VERSION} /usr/bin/python \
      && rm -rf /var/lib/apt/lists/* \
      && command -v ibv_devices; \
    fi

# Always refresh RDMA userspace on reuse builds so older tags cannot keep a
# stale verbs stack.
RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-change-held-packages \
      libibverbs1 \
      libibverbs-dev \
      ibverbs-providers \
      rdma-core \
      librdmacm-dev \
      libnuma-dev \
      numactl \
      infiniband-diags \
      ibverbs-utils \
 && (apt-get install -y --no-install-recommends libmlx5-1 librdmacm1 || true) \
 && rm -rf /var/lib/apt/lists/* \
 && command -v ibv_devices \
 && echo "Kimodo image RDMA userspace ready"

# Hanhai/Kubeflow Notebook SSH account (jovyan).
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100
ENV NB_USER=${NB_USER} \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID}
RUN mkdir -p /run/sshd /etc/ssh/sshd_config.d \
 && rm -f /etc/ssh/ssh_host_* \
 && printf '%s\n' \
      'PasswordAuthentication no' \
      'KbdInteractiveAuthentication no' \
      'PubkeyAuthentication yes' \
      'PermitRootLogin prohibit-password' \
      'AllowTcpForwarding yes' \
      > /etc/ssh/sshd_config.d/kimodo.conf \
 && if ! getent group "${NB_GID}" >/dev/null; then \
      groupadd -g "${NB_GID}" users || groupadd -g "${NB_GID}" "${NB_USER}"; \
    fi \
 && if ! getent passwd "${NB_USER}" >/dev/null; then \
      if getent passwd "${NB_UID}" >/dev/null; then \
        echo "NB_UID=${NB_UID} already present; leaving existing account in place"; \
      else \
        useradd -M -u "${NB_UID}" -g "${NB_GID}" -d "/home/${NB_USER}" -s /bin/bash "${NB_USER}"; \
      fi; \
    fi \
 && mkdir -p "/home/${NB_USER}" \
 && chown "${NB_UID}:${NB_GID}" "/home/${NB_USER}" \
 && if getent passwd "${NB_USER}" >/dev/null; then \
      usermod -p '*' "${NB_USER}"; \
    fi

EXPOSE 22

RUN rm -f /usr/local/bin/cmake || true

COPY docker_requirements.txt /workspace/docker_requirements.txt
COPY setup.py /workspace/setup.py
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY kimodo /workspace/kimodo
COPY MotionCorrection /workspace/MotionCorrection

# Bootstrap pip + PyTorch cu130 (same versions/index as cluster-proven images).
# Keep torch and Kimodo requirements in separate layers so PyPI flakes do not
# force re-downloading multi-GB CUDA wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${KIMODO_REUSE_BASE_ENV}" = 1 ]; then \
      python -c "import torch; print('reusing torch:', torch.__version__, torch.version.cuda)"; \
    else \
      wget -q https://bootstrap.pypa.io/get-pip.py \
      && python get-pip.py \
      && rm -f get-pip.py \
      && pip install --upgrade "pip==24.2" \
      && pip install --retries 10 \
           "torch==${KIMODO_TORCH_VERSION}" \
           "torchvision==${KIMODO_TORCHVISION_VERSION}" \
           "torchaudio==${KIMODO_TORCHAUDIO_VERSION}" \
           --index-url https://download.pytorch.org/whl/cu130 \
      && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${KIMODO_REUSE_BASE_ENV}" = 1 ]; then \
      python -c "import kimodo, torch, wandb; print('reusing validated environment:', torch.__version__, wandb.__version__)"; \
    else \
      # Training does not need MotionCorrection C++. Building it under QEMU/amd64
      # on Apple Silicon hits a GCC LTO ICE; install deps without it, then try a
      # no-LTO build and allow skip if that still fails.
      grep -vE '^\./MotionCorrection[[:space:]]*$' docker_requirements.txt \
        > /tmp/docker_requirements.nomc.txt \
      && PIP_NO_CACHE_DIR=0 SKIP_MOTION_CORRECTION_IN_SETUP=1 \
           pip install --retries 10 \
             -i https://mirrors.aliyun.com/pypi/simple/ \
             --trusted-host mirrors.aliyun.com \
             -r /tmp/docker_requirements.nomc.txt \
      && if CMAKE_ARGS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF -DCMAKE_CXX_FLAGS=-fno-lto" \
            CFLAGS="-fno-lto" CXXFLAGS="-fno-lto" \
            pip install --retries 5 --no-build-isolation ./MotionCorrection; then \
           echo "Kimodo image: MotionCorrection C++ ext installed"; \
         else \
           echo "Kimodo image: optional MotionCorrection C++ ext skipped (training OK)"; \
         fi \
      && python -c "import kimodo, torch, wandb; print('kimodo image smoke:', torch.__version__, torch.version.cuda, 'wandb', wandb.__version__)"; \
    fi

RUN find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
COPY . /workspace
RUN find /workspace/scripts -type f -name '*.sh' -exec chmod +x {} + \
 && git config --system --add safe.directory /workspace \
 && chmod -R a+rwX /workspace

ARG KIMODO_SKIP_SMOKE=0
RUN python -m kimodo.evaluation.eval_monitor_cli --help >/dev/null \
 && python -m kimodo.resources.cli --help >/dev/null
RUN --mount=type=tmpfs,target=/tmp \
    if [ "${KIMODO_SKIP_SMOKE}" = 1 ]; then \
      echo "Kimodo image: skipping smoke_train (KIMODO_SKIP_SMOKE=1)"; \
    else \
      KIMODO_CODE_ROOT=/workspace \
      KIMODO_STORAGE_ROOT=/tmp/kimodo-smoke-storage \
      KIMODO_PYTHON=python /workspace/scripts/smoke_train.sh; \
    fi

COPY kimodo/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

ARG KIMODO_GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=${KIMODO_GIT_COMMIT}
ENV KIMODO_IMAGE_GIT_COMMIT=${KIMODO_GIT_COMMIT}

ENTRYPOINT ["docker-entrypoint"]
CMD ["/workspace/scripts/container_start.sh"]
