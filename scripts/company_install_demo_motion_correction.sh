#!/usr/bin/env bash
# Build MotionCorrection into the kimodo-dev demo venv.
# kimodo-dev has no system g++. Use already-extracted conda-forge compiler
# packages under /opt/conda/pkgs (do not conda create: 16GB cgroup OOMs).
# Do not run this on the 16-GPU trainer.
set -euo pipefail

export KIMODO_CODE_ROOT="${KIMODO_CODE_ROOT:-/home/share/yzt/kimodo-reproduction}"
DEMO_PYTHON="${KIMODO_DEMO_PYTHON:-/home/jovyan/.venv-kimodo-demo/bin/python}"
SRC="${KIMODO_CODE_ROOT}/MotionCorrection"
SITE="$("${DEMO_PYTHON}" -c "import site; print(site.getsitepackages()[0])")"

if [[ ! -x "${DEMO_PYTHON}" ]]; then
  echo "demo python missing: ${DEMO_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${SRC}/CMakeLists.txt" ]]; then
  echo "MotionCorrection missing: ${SRC}" >&2
  exit 2
fi

if [[ -z "${http_proxy:-}${HTTP_PROXY:-}" ]] && ss -lnt 2>/dev/null | grep -q ':7993'; then
  export http_proxy="${KIMODO_DEMO_HTTP_PROXY:-http://127.0.0.1:7993}"
  export https_proxy="${https_proxy:-${http_proxy}}"
  export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
  export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
fi

if "${DEMO_PYTHON}" -c "from motion_correction import motion_postprocess" 2>/dev/null; then
  echo "motion_correction already importable"
  "${DEMO_PYTHON}" -c "import motion_correction; print(motion_correction.__file__)"
  exit 0
fi

if command -v g++ >/dev/null && command -v cmake >/dev/null && command -v ninja >/dev/null; then
  "${DEMO_PYTHON}" -m pip install -U pip pybind11
  "${DEMO_PYTHON}" -m pip install --no-build-isolation -e "${SRC}"
else
  GCCPK=/opt/conda/pkgs/gcc_impl_linux-64-15.3.0-h3a305e7_1
  GXXPK=/opt/conda/pkgs/gxx_impl_linux-64-15.3.0-h90d9265_1
  BINU=/opt/conda/pkgs/binutils_impl_linux-64-2.46.1-default_hfdba357_102/bin
  CRT=/opt/conda/pkgs/libgcc-devel_linux-64-15.3.0-h2b852eb_101/lib/gcc/x86_64-conda-linux-gnu/15.3.0
  LIBGCC=/opt/conda/pkgs/libgcc-16.1.0-ha9f2e26_1/lib
  LIBSTD=/opt/conda/pkgs/libstdcxx-16.1.0-h934c35e_1/lib
  LIBSTDDEV=/opt/conda/pkgs/libstdcxx-devel_linux-64-15.3.0-hb2c5482_101/lib
  CXXINC=${LIBSTDDEV}/gcc/x86_64-conda-linux-gnu/15.3.0/include/c++
  GCCINC=${GCCPK}/lib/gcc/x86_64-conda-linux-gnu/15.3.0/include
  SYSROOT=/opt/conda/pkgs/sysroot_linux-64-2.34-h087de78_3/x86_64-conda-linux-gnu/sysroot
  KERNEL=/opt/conda/pkgs/kernel-headers_linux-64-5.14.0-he073ed8_3/x86_64-conda-linux-gnu/sysroot/usr/include
  EIGEN_DIR=/opt/conda/pkgs/eigen-5.0.1-hc65338a_0/share/eigen3/cmake
  TOOLS=/tmp/mc-tools
  mkdir -p "${TOOLS}"
  ln -sfn "${BINU}/x86_64-conda-linux-gnu-as" "${TOOLS}/as"
  ln -sfn "${BINU}/x86_64-conda-linux-gnu-ld.bfd" "${TOOLS}/ld"
  ln -sfn "${BINU}/x86_64-conda-linux-gnu-ar" "${TOOLS}/ar"
  ln -sfn "${BINU}/x86_64-conda-linux-gnu-ranlib" "${TOOLS}/ranlib"
  ln -sfn "${BINU}/x86_64-conda-linux-gnu-nm" "${TOOLS}/nm"
  ln -sfn "${BINU}/x86_64-conda-linux-gnu-strip" "${TOOLS}/strip"
  export CONDA_BUILD_SYSROOT="${SYSROOT}"
  export CC="${GCCPK}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${GXXPK}/bin/x86_64-conda-linux-gnu-g++"
  export PATH="${TOOLS}:${HOME}/.venv-kimodo-demo/bin:${PATH}"
  export LIBRARY_PATH="${CRT}:${LIBGCC}:${LIBSTD}:${LIBSTDDEV}"
  export CPLUS_INCLUDE_PATH="${CXXINC}:${CXXINC}/x86_64-conda-linux-gnu:${GCCINC}:${KERNEL}"
  export C_INCLUDE_PATH="${GCCINC}:${KERNEL}"
  FLAGS="--sysroot=${SYSROOT} -fno-use-linker-plugin -isystem ${KERNEL}"
  export CFLAGS="${FLAGS}" CXXFLAGS="${FLAGS}"
  PYBIND="$("${DEMO_PYTHON}" -c "import pybind11; print(pybind11.get_cmake_dir())")"
  BUILD=/tmp/motion-correction-build
  rm -rf "${BUILD}"
  mkdir -p "${BUILD}"
  cmake "${SRC}" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$(command -v ninja)" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${FLAGS}" \
    -DCMAKE_CXX_FLAGS="${FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${FLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${FLAGS}" \
    -DCMAKE_SYSROOT="${SYSROOT}" \
    -DPython3_EXECUTABLE="${DEMO_PYTHON}" \
    -DPython3_INCLUDE_DIR=/opt/conda/include/python3.11 \
    -Dpybind11_DIR="${PYBIND}" \
    -DEigen3_DIR="${EIGEN_DIR}" \
    -DCMAKE_PREFIX_PATH=/opt/conda/pkgs/eigen-5.0.1-hc65338a_0 \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${SRC}/python/motion_correction" \
    -B "${BUILD}"
  ninja -C "${BUILD}"
  mkdir -p "${SITE}/motion_correction"
  cp -a "${SRC}/python/motion_correction/." "${SITE}/motion_correction/"
fi

"${DEMO_PYTHON}" - <<'PY'
import motion_correction
from motion_correction import motion_postprocess
print("motion_correction_ok", getattr(motion_correction, "__file__", motion_correction))
print("postprocess_ok", motion_postprocess)
PY
