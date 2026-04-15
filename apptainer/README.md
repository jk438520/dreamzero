# Apptainer: interactive GPU shell (NGC PyTorch 26.03)

This folder contains a small helper script to open an **interactive** Apptainer shell using the NVIDIA NGC container:

- `nvcr.io/nvidia/pytorch:26.03-py3`

It bind-mounts your repo at the **same path** inside the container:

- `~/jakkol/dreamzero` → `~/jakkol/dreamzero`

That usually avoids breaking symlinks inside the repo.

It also bind-mounts your cache directory by default:

- `$XDG_CACHE_HOME` (if set) or `~/.cache`

This is important when repo folders symlink into `~/.cache/...`.

## 0) Prerequisites

- Apptainer installed on the cluster (often via modules)
- NVIDIA GPUs + drivers available (for `--nv`)

Typical cluster setup (example):

```bash
module load apptainer
# or: module load singularity
```

Verify:

```bash
apptainer --version
```

## 1) Quick start (ephemeral installs)

This mode lets you `pip install` / `apt-get install` interactively, but **changes are lost when you exit**.

```bash
cd ~/jakkol/dreamzero/apptainer
chmod +x run_pytorch_26_03_shell.sh
./run_pytorch_26_03_shell.sh
```

Inside the container you should land in:

```bash
pwd
# ~/jakkol/dreamzero
```

## 2) Persistent installs (overlay)

Apptainer containers are read-only by default. To keep installed packages across sessions, use an **overlay** file.

### Create an overlay file (once)

Pick a location in your home directory:

```bash
mkdir -p "$HOME/.apptainer/overlays"

# Create a 20GB ext3 overlay file (adjust size as needed)
apptainer overlay create --size 20480 "$HOME/.apptainer/overlays/pytorch2603.ext3"
```

### Use the overlay

```bash
cd ~/jakkol/dreamzero/apptainer
./run_pytorch_26_03_shell.sh --overlay "$HOME/.apptainer/overlays/pytorch2603.ext3"
```

Now installs you do inside the container will persist in the overlay file.

## 3) Notes / troubleshooting

### Using conda inside the container (recommended: micromamba)

Conda-style envs can be very convenient for GPU stacks, but you generally want them **isolated** (and stored in a high-quota location).

1) First, check if the NGC image already ships conda:

```bash
conda --version || true
which conda || true
```

If `conda` exists, you can create envs in your cache (host-persistent):

```bash
mkdir -p "$HOME/.cache/jakkol/dreamzero/conda_envs"
conda create -y -p "$HOME/.cache/jakkol/dreamzero/conda_envs/dz" python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.cache/jakkol/dreamzero/conda_envs/dz"
```

2) If `conda` is NOT present, the simplest approach is **micromamba** (single binary; fast; no root):

```bash
# Install micromamba into your host cache (so it persists across sessions)
mkdir -p "$HOME/.cache/jakkol/dreamzero/bin"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C "$HOME/.cache/jakkol/dreamzero/bin" --strip-components=1 bin/micromamba

MAMBA="$HOME/.cache/jakkol/dreamzero/bin/micromamba"

# Create an env in cache (host-persistent)
mkdir -p "$HOME/.cache/jakkol/dreamzero/conda_envs"
"$MAMBA" create -y -p "$HOME/.cache/jakkol/dreamzero/conda_envs/dz" python=3.10

# Activate for the current shell session (no conda-init needed)
eval "$("$MAMBA" shell hook -s bash)"
micromamba activate "$HOME/.cache/jakkol/dreamzero/conda_envs/dz"
```

Notes:

- Avoid `conda init` inside the container: Apptainer typically exposes your real `$HOME`, so `conda init` would edit your host shell rc files.
- Conda envs can get large; if your cluster has `/scratch/$USER`, storing envs there is often better than `~/.cache`.
- If your cluster blocks outbound `curl`, you may need to download micromamba on a login node that has internet, or ask your admins.

### Saving large checkpoints (e.g. 50GB)

For large artifacts, you usually **don’t** want them inside the container filesystem or overlay.

This launcher bind-mounts your cache dir (`$XDG_CACHE_HOME` or `~/.cache`) from the host, so anything you write there is saved on the host directly.

Example: save directly to your host cache from inside the container:

```bash
mkdir -p "$HOME/.cache/jakkol/dreamzero/checkpoints"
export DREAMZERO_CKPT_DIR="$HOME/.cache/jakkol/dreamzero/checkpoints"
```

Then point your training script to `$DREAMZERO_CKPT_DIR` (exact flag/env var depends on the script).

If you want your code to keep using the path `~/jakkol/dreamzero/checkpoints`, you can make `checkpoints/` a symlink to cache on the **host**:

```bash
cd ~/jakkol/dreamzero
mkdir -p "$HOME/.cache/jakkol/dreamzero/checkpoints"

# If you already have a real directory, move it first (optional)
test -d checkpoints && test ! -L checkpoints && mv checkpoints checkpoints.bak

ln -sfn "$HOME/.cache/jakkol/dreamzero/checkpoints" checkpoints
```

Because `~/.cache` is bind-mounted, the symlink will resolve inside the container too.

If your cluster provides a high-quota filesystem (often `/scratch/$USER`), that’s usually even better for 50GB+ outputs; just bind it and write there.

### Reserving only 2 GPUs

Apptainer does **not** reserve GPUs by itself — your cluster scheduler does. The typical workflow is:

1) Request an interactive allocation with 2 GPUs
2) Run the Apptainer shell inside that allocation

If your scheduler already sets `CUDA_VISIBLE_DEVICES`, the container will usually only see the allocated GPUs.

Optionally, you can also limit visibility yourself (this does **not** reserve GPUs):

```bash
./run_pytorch_26_03_shell.sh --gpus 0,1
```

### GPU not detected

- Make sure you used `--nv` (default in the script)
- Check `nvidia-smi` on the host (outside the container)

Inside the container, you can sanity check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Symlinks inside `~/jakkol/dreamzero`

Bind-mounting at the same path helps, but symlinks can still break if they point **outside** `~/jakkol/dreamzero`.

If a symlink points to some other host path (e.g. `~/datasets/...` or `/scratch/...`), you must also bind that target into the container. You can pass extra bind mounts after `--`:

```bash
./run_pytorch_26_03_shell.sh -- \
  --bind "$HOME/datasets:$HOME/datasets" \
  --bind "/scratch/$USER:/scratch/$USER"
```

The launcher script also tries to help here: it will automatically bind-mount the targets of **top-level symlinks** in `~/jakkol/dreamzero`.

For example, if you have:

- `~/jakkol/dreamzero/data -> ~/.cache/jakkol/dreamzero/data`

it will add a bind mount for that cache path so `data/` works inside the container.

If you *don’t* want this behavior, launch with:

```bash
./run_pytorch_26_03_shell.sh --no-auto-bind-symlinks
```

If you need to disable cache binding (rare), use:

```bash
./run_pytorch_26_03_shell.sh --no-bind-cache
```

### Network / proxy / SSL issues

Clusters sometimes require proxies or custom certificates. Because the script uses `--cleanenv`, your proxy env vars may be cleared.

If you need proxies, you can pass them explicitly when launching:

```bash
HTTPS_PROXY=http://... HTTP_PROXY=http://... ./run_pytorch_26_03_shell.sh
```

Or remove `--cleanenv` from the script (less isolated, but more convenient).
