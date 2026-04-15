#!/usr/bin/env bash
set -euo pipefail

IMAGE_URI_DEFAULT="docker://nvcr.io/nvidia/pytorch:26.03-py3"

usage() {
  cat <<'EOF'
Usage:
  ./run_pytorch_26_03_shell.sh [--overlay <path>] [--gpus <list>] [--no-nv] [--image <uri>] [--no-auto-bind-symlinks] [--no-bind-cache] [--] [extra apptainer args...]

Starts an interactive Apptainer shell using NVIDIA's PyTorch NGC container.

Defaults:
  - Image: docker://nvcr.io/nvidia/pytorch:26.03-py3
  - GPU: enabled (--nv)
  - Your repo bind-mounted at the SAME PATH inside the container:
      $HOME/jakkol/dreamzero  ->  $HOME/jakkol/dreamzero
    This preserves symlinks that assume that path.
  - Working directory set to: $HOME/jakkol/dreamzero

Writable behavior:
  - By default uses --writable-tmpfs (installs are ephemeral; lost when you exit).
  - With --overlay <path>, installs persist in the overlay file.

Examples:
  # Ephemeral installs (good for quick experimentation)
  ./run_pytorch_26_03_shell.sh

  # Persistent installs via overlay image
  ./run_pytorch_26_03_shell.sh --overlay "$HOME/.apptainer/overlays/pytorch2603.ext3"

  # Limit *visibility* to 2 GPUs inside the container (does NOT reserve GPUs)
  ./run_pytorch_26_03_shell.sh --gpus 0,1

  # If your cluster forbids --nv or you want CPU-only
  ./run_pytorch_26_03_shell.sh --no-nv

  # Disable auto-binding symlink targets
  ./run_pytorch_26_03_shell.sh --no-auto-bind-symlinks

  # Disable binding your cache dir (~/.cache)
  ./run_pytorch_26_03_shell.sh --no-bind-cache
EOF
}

APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"

if ! command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$APPTAINER_BIN' not found in PATH. Load your module (e.g. 'module load apptainer') or install Apptainer." >&2
  exit 127
fi

IMAGE_URI="$IMAGE_URI_DEFAULT"
USE_NV=1
OVERLAY_PATH=""
GPU_LIST=""
AUTO_BIND_SYMLINKS=1
BIND_CACHE=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --image)
      IMAGE_URI="$2"
      shift 2
      ;;
    --overlay)
      OVERLAY_PATH="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --no-auto-bind-symlinks)
      AUTO_BIND_SYMLINKS=0
      shift
      ;;
    --no-bind-cache)
      BIND_CACHE=0
      shift
      ;;
    --no-nv)
      USE_NV=0
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

HOST_REPO_DIR="${HOST_REPO_DIR:-$HOME/jakkol/dreamzero}"

if [[ ! -d "$HOST_REPO_DIR" ]]; then
  echo "ERROR: Host repo dir not found: $HOST_REPO_DIR" >&2
  echo "Set HOST_REPO_DIR to override, e.g.:" >&2
  echo "  HOST_REPO_DIR=\"$PWD\" ./run_pytorch_26_03_shell.sh" >&2
  exit 2
fi

# Keep the same path inside the container to avoid breaking symlinks.
CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-$HOST_REPO_DIR}"

APPTAINER_ARGS=(
  shell
  --cleanenv
  --pwd "$CONTAINER_REPO_DIR"
  --bind "$HOST_REPO_DIR:$CONTAINER_REPO_DIR"
)

if [[ $BIND_CACHE -eq 1 ]]; then
  CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}"
  if [[ -d "$CACHE_DIR" ]]; then
    APPTAINER_ARGS+=(--bind "$CACHE_DIR:$CACHE_DIR")
  fi
fi

if [[ -n "$GPU_LIST" ]]; then
  # NOTE: This only limits *visibility* inside the container.
  # Actual GPU reservation must be done via the scheduler (Slurm/PBS/etc.).
  APPTAINER_ARGS+=(--env "CUDA_VISIBLE_DEVICES=$GPU_LIST")
  APPTAINER_ARGS+=(--env "NVIDIA_VISIBLE_DEVICES=$GPU_LIST")
fi

if [[ $AUTO_BIND_SYMLINKS -eq 1 ]]; then
  # Many repos keep large artifacts outside the repo and symlink them in (e.g. data -> ~/.cache/...).
  # Those symlink targets must be bind-mounted too, otherwise they appear broken inside the container.
  #
  # To keep this predictable, we only scan and auto-bind symlinks at the repo root (maxdepth 1).
  declare -A _BIND_SEEN=()
  while IFS= read -r -d '' link_path; do
    # Resolve the symlink to an absolute, canonical path on the host.
    target_abs="$(readlink -f "$link_path" 2>/dev/null || true)"
    if [[ -z "$target_abs" ]]; then
      continue
    fi

    # Only bind if the target exists on the host.
    if [[ -e "$target_abs" ]]; then
      if [[ -z "${_BIND_SEEN[$target_abs]+x}" ]]; then
        APPTAINER_ARGS+=(--bind "$target_abs:$target_abs")
        _BIND_SEEN["$target_abs"]=1
      fi
    else
      echo "WARN: symlink target missing on host: $link_path -> $target_abs" >&2
    fi
  done < <(find "$HOST_REPO_DIR" -maxdepth 1 -type l -print0 2>/dev/null || true)
fi

if [[ $USE_NV -eq 1 ]]; then
  APPTAINER_ARGS+=(--nv)
fi

if [[ -n "$OVERLAY_PATH" ]]; then
  # Create parent dir if needed; overlay file must already exist.
  if [[ ! -f "$OVERLAY_PATH" ]]; then
    echo "ERROR: Overlay file does not exist: $OVERLAY_PATH" >&2
    echo "Create one first (see apptainer/README.md)." >&2
    exit 2
  fi
  APPTAINER_ARGS+=(--overlay "$OVERLAY_PATH")
else
  # Ephemeral writes in RAM/tmp; good for interactive pip/apt installs without touching the host.
  APPTAINER_ARGS+=(--writable-tmpfs)
fi

APPTAINER_ARGS+=("${EXTRA_ARGS[@]}")
APPTAINER_ARGS+=("$IMAGE_URI")

echo "==> Apptainer: $APPTAINER_BIN"
echo "==> Image:     $IMAGE_URI"
echo "==> GPU:       $([[ $USE_NV -eq 1 ]] && echo enabled || echo disabled)"
if [[ -n "$GPU_LIST" ]]; then
  echo "==> CUDA_VISIBLE_DEVICES: $GPU_LIST"
  echo "==> NVIDIA_VISIBLE_DEVICES: $GPU_LIST"
fi
echo "==> Repo bind: $HOST_REPO_DIR -> $CONTAINER_REPO_DIR"
if [[ -n "$OVERLAY_PATH" ]]; then
  echo "==> Overlay:   $OVERLAY_PATH (persistent)"
else
  echo "==> Overlay:   none (using --writable-tmpfs; ephemeral)"
fi

echo "==> Starting interactive shell..."
exec "$APPTAINER_BIN" "${APPTAINER_ARGS[@]}"
