#!/bin/bash
# First-time cluster setup.
# MUST be run on a compute node, not the login node:
#   srun -p u22 --gres=gpu:1 -c 4 --time=00:30:00 --pty bash setup.sh
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "=== to-fa setup ==="
echo "Project: $PROJECT_DIR  |  node=$(hostname)"

# Scratch is /scratch/$USER (shared NFS, not node-local)
export SCRATCH="/scratch/$USER"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export PATH="$HOME/.pixi/bin:$PATH"

# 1. Install pixi if missing
if ! command -v pixi &>/dev/null; then
    echo "[1/4] Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
else
    echo "[1/4] Pixi already installed"
fi

# 2. Create scratch dirs (only works on compute node)
echo "[2/4] Creating scratch dirs at $SCRATCH ..."
mkdir -p "$SCRATCH"/{hf_cache/hub,outputs,checkpoints,experiments}
pixi config set detached-environments "$SCRATCH/pixi-envs"
echo "  Storage ready."

# 3. Install project deps
echo "[3/4] Installing project deps..."
cd "$PROJECT_DIR"
pixi install

# 4. Write env vars to ~/.bashrc (login node uses these for non-job work)
echo "[4/4] Updating ~/.bashrc..."
if ! grep -q "to-fa scratch" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

# to-fa scratch — node-local path set at login; overridden per-job in serve.slurm
export PATH="\$HOME/.pixi/bin:\$PATH"
# to-fa scratch
EOF
    echo "  Updated ~/.bashrc"
else
    echo "  ~/.bashrc already configured"
fi

echo ""
echo "Done. Scratch: $SCRATCH"
echo "Exit this srun session, then: ./sync.sh && sbatch serve.slurm"
