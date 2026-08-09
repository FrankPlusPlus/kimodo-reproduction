# syntax=docker/dockerfile:1.7

# Keep the known project base as the safe default. Company CI may override it
# with an approved, digest-pinned NGC image after checking the host driver.
ARG KIMODO_PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.10-py3
FROM ${KIMODO_PYTORCH_IMAGE}

# Company rebuilds may layer a source-only update on the already validated v2
# image. The default remains a clean NGC bootstrap for independent CI builds.
ARG KIMODO_REUSE_BASE_ENV=0

# Avoid some interactive prompts + make pip quieter/reproducible-ish
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KIMODO_CODE_ROOT=/home/share/yzt/kimodo-reproduction \
    KIMODO_STORAGE_ROOT=/home/share/yezitao-kimodo-reproduction

# Build/install workdir. At runtime, reviewed modes prefer KIMODO_CODE_ROOT on
# the share PVC; /workspace remains the image-local fallback for idle/smoke.
WORKDIR /workspace

# System deps
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
        git curl ca-certificates \
        cmake build-essential \
        gosu \
        zstd openssh-server \
        libibverbs1 ibverbs-providers rdma-core \
        infiniband-diags perftest ibutils ibverbs-utils \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Hanhai/Kubeflow Notebook SSH: gateway authenticates user-ssh-public-key, then
# forwards to Pod :22 and logs in as jovyan (the Jupyter/Kubeflow default login
# account; home is /home/jovyan, usually the Notebook workspace PVC). Custom
# images must run sshd, ship that account, and install authorized_keys at
# runtime from KIMODO_SSH_PUBLIC_KEY or a mounted file.
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
      # useradd leaves "!" locked accounts; OpenSSH rejects locked users even for
      # publickey. "*" disables password auth while keeping the account usable.
      usermod -p '*' "${NB_USER}"; \
    fi

EXPOSE 22

# Some base images ship a broken `/usr/local/bin/cmake` shim (from a partial pip install),
# which shadows `/usr/bin/cmake` and breaks builds that invoke `cmake` (e.g. MotionCorrection).
# Prefer the system cmake.
RUN rm -f /usr/local/bin/cmake || true

# Install from docker_requirements.txt: kimodo editable (-e .),
# but MotionCorrection non-editable (./MotionCorrection). The -e . line ensures [project.scripts]
# from pyproject.toml are installed (kimodo_gen, kimodo_demo, kimodo_textencoder).
# SKIP_MOTION_CORRECTION_IN_SETUP=1 so setup.py does not bundle motion_correction; it is
# installed separately from ./MotionCorrection in the requirements file (non-editable).
# The interactive demo's optional kimodo-viser dependency is intentionally not
# part of this train/eval image.
COPY docker_requirements.txt /workspace/docker_requirements.txt
COPY setup.py /workspace/setup.py
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY kimodo /workspace/kimodo
COPY MotionCorrection /workspace/MotionCorrection

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${KIMODO_REUSE_BASE_ENV}" = 1 ]; then \
      python -c "import kimodo, torch, wandb; print('reusing validated environment:', torch.__version__, wandb.__version__)"; \
    else \
      PIP_NO_CACHE_DIR=0 python -m pip install --upgrade "pip==24.2" \
      && PIP_NO_CACHE_DIR=0 SKIP_MOTION_CORRECTION_IN_SETUP=1 python -m pip install -r docker_requirements.txt; \
    fi

# NGC 24.10 contains a legacy two-field wheel tag that makes `pip check`
# crash while scanning platform compatibility. Check missing/version conflicts
# directly and skip only that broken tag scan.
RUN python -c "from pip._internal.operations.check import create_package_set_from_installed, check_package_set; package_set, parsing_problems = create_package_set_from_installed(); missing, conflicting = check_package_set(package_set, should_ignore=lambda _name: False); print('dependency check:', len(missing), 'missing,', len(conflicting), 'conflicting'); assert not parsing_problems and not missing and not conflicting, (parsing_problems, missing, conflicting)" \
 && python -c "import kimodo, torch, wandb; print('kimodo image smoke:', torch.__version__, torch.version.cuda, 'wandb', wandb.__version__)"

# Copy the complete Git working tree after the dependency layer. The image still
# carries /workspace for dependency install, smoke tests, and idle fallback.
# Company training reads code from KIMODO_CODE_ROOT on the share PVC
# (/home/share/yzt/kimodo-reproduction). Datasets/checkpoints stay on
# KIMODO_STORAGE_ROOT and are excluded by .dockerignore.
RUN find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
COPY . /workspace
RUN find /workspace/scripts -type f -name '*.sh' -exec chmod +x {} + \
 && git config --system --add safe.directory /workspace \
 && chmod -R a+rwX /workspace

# Fail the image build if the post-migration training/evaluation entry points
# or the real two-step trainer path are broken. The generated fixture, logs and
# checkpoints live only on a BuildKit tmpfs and are not retained in the image.
RUN python -m kimodo.evaluation.eval_monitor_cli --help >/dev/null \
 && python -m kimodo.resources.cli --help >/dev/null
RUN --mount=type=tmpfs,target=/tmp \
    KIMODO_CODE_ROOT=/workspace \
    KIMODO_STORAGE_ROOT=/tmp/kimodo-smoke-storage \
    KIMODO_PYTHON=python /workspace/scripts/smoke_train.sh

# Use the docker-entrypoint script, to allow the docker to run as the actual user instead of root
COPY kimodo/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint
RUN chmod +x /usr/local/bin/docker-entrypoint

# Keep source revision metadata late in the file so changing only the Git
# commit does not invalidate apt and Python dependency layers.
ARG KIMODO_GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=${KIMODO_GIT_COMMIT}
ENV KIMODO_IMAGE_GIT_COMMIT=${KIMODO_GIT_COMMIT}

# Keep the image hardware-neutral by default.  The dispatcher stays alive in
# idle mode, while Kubernetes can either override the command with a concrete
# launcher or select an explicit KIMODO_CONTAINER_MODE.  Merely creating a Pod
# must not require 16 H200s or RDMA; train modes expect the share PVC at
# /home/share with code under KIMODO_CODE_ROOT.
ENTRYPOINT ["docker-entrypoint"]
CMD ["/workspace/scripts/container_start.sh"]
