#!/usr/bin/env bash
# Run the complete deployment on a Jetson with a wheel set matching its L4T.
set -euo pipefail

readonly MANAGED_BASE_URL="https://f005.backblazeb2.com/file/jetson-assets"
readonly DEFAULT_RELEASE_FILE="/etc/nv_tegra_release"

die() {
  echo "jetson-compose: $*" >&2
  exit 1
}

parse_l4t_version() {
  local contents="$1"
  if [[ "$contents" =~ R([0-9]+).*REVISION:[[:space:]]*([0-9.]+) ]]; then
    printf '%s.%s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]%.}"
  else
    return 1
  fi
}

set_wheel_urls() {
  # Arguments: package filename on the managed mirror, then zero or more fallbacks.
  local managed_key="$1"
  shift
  local urls=()
  if [[ -n "$managed_key" ]]; then
    urls+=("${MANAGED_BASE_URL}/${managed_key}")
  fi
  urls+=("$@")
  printf '%s' "${urls[*]}"
}

resolve_jetson() {
  local l4t_version="$1"
  export JETSON_PYTHON="python3"

  case "$l4t_version" in
    36.2*|36.3*)
      export JETPACK_VERSION="6.0"
      export L4T_BASE="nvcr.io/nvidia/l4t-jetpack:r36.2.0"
      export CUDA_VERSION="12.2"
      export NUMPY_VERSION="1.26.1"
      export CUSPARSELT_REQUIRED="0"
      export TORCH_WHEEL_FILENAME="torch-2.3.0-cp310-cp310-linux_aarch64.whl"
      export TORCH_WHEEL_URLS
      TORCH_WHEEL_URLS="$(set_wheel_urls \
        'jetson-wheels/jp6/cu122/torch-2.3.0-cp310-cp310-linux_aarch64.whl' \
        'https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl')"
      export TORCHVISION_WHEEL_URLS
      export TORCHVISION_WHEEL_FILENAME="torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"
      TORCHVISION_WHEEL_URLS="$(set_wheel_urls \
        'jetson-wheels/jp6/cu122/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl' \
        'https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl')"
      ;;
    36.4.3*|36.4.4*|36.5.0*|36.5.2*)
      export JETPACK_VERSION="6.2"
      # NVIDIA's latest published L4T JetPack image is r36.4.0; its CUDA
      # 12.6 userspace is compatible with the launcher wheel family used by
      # the r36.4/r36.5 JetPack 6.2 releases.
      export L4T_BASE="nvcr.io/nvidia/l4t-jetpack:r36.4.0"
      export CUDA_VERSION="12.6"
      export NUMPY_VERSION="1.26.1"
      export CUSPARSELT_REQUIRED="1"
      export TORCH_WHEEL_FILENAME="torch-2.8.0-cp310-cp310-linux_aarch64.whl"
      export TORCH_WHEEL_URLS
      TORCH_WHEEL_URLS="$(set_wheel_urls \
        'jetson-wheels/jp6/cu126/torch-2.8.0-cp310-cp310-linux_aarch64.whl' \
        'https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc')"
      export TORCHVISION_WHEEL_URLS
      export TORCHVISION_WHEEL_FILENAME="torchvision-0.23.0-cp310-cp310-linux_aarch64.whl"
      TORCHVISION_WHEEL_URLS="$(set_wheel_urls \
        'jetson-wheels/jp6/cu126/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl' \
        'https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1')"
      ;;
    39.2.1*)
      export JETPACK_VERSION="7.2"
      # L4T containers are no longer published for JP7.  The NVIDIA Ubuntu
      # ARM64 base receives the host CUDA stack from the NVIDIA runtime.
      export L4T_BASE="nvcr.io/nvidia/base/ubuntu:24.04"
      export CUDA_VERSION="13.2"
      export NUMPY_VERSION=""
      export CUSPARSELT_REQUIRED="0"
      export TORCH_WHEEL_FILENAME="torch-2.14.1+cu132-cp312-cp312-manylinux_2_28_aarch64.whl"
      # JetPack 7.2 is the launcher exception: its manifest publishes only
      # upstream PyTorch URLs, so there is no managed-mirror artifact to try.
      export TORCH_WHEEL_URLS='https://download-r2.pytorch.org/whl/cu132/torch-2.14.1%2Bcu132-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=2ac1397f72096bd31740e2f0c2d9477111d32dbcec54e1f308063365a9dcc7b5'
      export TORCHVISION_WHEEL_URLS='https://download-r2.pytorch.org/whl/cu132/torchvision-0.29.1%2Bcu132-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=fa9f736052a7350575a64079150932da6d130d6cb6a704956162df40b3c7e4e9'
      export TORCHVISION_WHEEL_FILENAME="torchvision-0.29.1+cu132-cp312-cp312-manylinux_2_28_aarch64.whl"
      ;;
    36.4*)
      die "JetPack 6.1 (L4T ${l4t_version}) has no launcher-verified torchvision wheel. Upgrade to JetPack 6.2 or newer."
      ;;
    35.*|34.*|32.*)
      die "JetPack 5 and older (L4T ${l4t_version}) are unsupported: this app needs Python dependencies and torchvision wheels unavailable for that release. Upgrade to JetPack 6.2 or newer."
      ;;
    *)
      die "unsupported L4T ${l4t_version}. Supported releases are JetPack 6.0, 6.2/6.2.x, and 7.2/7.2.x."
      ;;
  esac
}

main() {
  local release_file="${JETSON_RELEASE_FILE:-$DEFAULT_RELEASE_FILE}"
  [[ -r "$release_file" ]] || die "cannot read ${release_file}; run this on a flashed Jetson or set JETSON_RELEASE_FILE for diagnostics."
  local l4t_version
  l4t_version="$(parse_l4t_version "$(<"$release_file")")" || die "could not parse L4T version from ${release_file}."
  resolve_jetson "$l4t_version"

  if [[ "${1:-}" == "--print-env" ]]; then
    printf 'L4T_VERSION=%s\nJETPACK_VERSION=%s\nL4T_BASE=%s\nCUDA_VERSION=%s\nNUMPY_VERSION=%s\nCUSPARSELT_REQUIRED=%s\nTORCH_WHEEL_FILENAME=%s\nTORCHVISION_WHEEL_FILENAME=%s\n' \
      "$l4t_version" "$JETPACK_VERSION" "$L4T_BASE" "$CUDA_VERSION" "$NUMPY_VERSION" "$CUSPARSELT_REQUIRED" "$TORCH_WHEEL_FILENAME" "$TORCHVISION_WHEEL_FILENAME"
    return
  fi

  export L4T_BASE JETSON_PYTHON NUMPY_VERSION TORCH_WHEEL_FILENAME TORCH_WHEEL_URLS TORCHVISION_WHEEL_FILENAME TORCHVISION_WHEEL_URLS CUSPARSELT_REQUIRED CUDA_VERSION
  exec docker compose -f compose.jetson.yaml "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
