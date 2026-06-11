#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --partition=u22
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%j.out

# /scratch is node-local (same as /tmp on compute nodes).
# /scratch/$USER is created by prior jobs; use it if present, else /tmp/$USER.
SCRATCH_CANDIDATES=("/scratch/$USER" "/tmp/$USER")
for d in "${SCRATCH_CANDIDATES[@]}"; do
    if mkdir -p "$d/hf_cache/hub" 2>/dev/null; then
        export SCRATCH="$d"
        break
    fi
done
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"

export PATH="$HOME/.pixi/bin:$PATH"
eval "$(pixi shell-hook)"

CFG=$(python3 -c "
import yaml, pathlib
c = yaml.safe_load(pathlib.Path('config.yaml').read_text())
v = c['vllm']
exp = '${EXPERIMENT:-}'
if exp and exp in c.get('experiments', {}):
    for k, val in c['experiments'][exp].get('vllm', {}).items():
        v[k] = val
print(v['hf_model'], v['served_name'], v['port'], v['max_model_len'], v['gpu_memory_utilization'], str(v['enable_prefix_caching']).lower(), v.get('extra_args', ''))
")
read HF_MODEL SERVED_NAME PORT MAX_LEN GPU_MEM PREFIX_CACHE EXTRA_ARGS <<< "$CFG"

echo "experiment=${EXPERIMENT:-baseline}  node=$(hostname)  job=$SLURM_JOB_ID"
echo "SCRATCH=$SCRATCH  HF_HOME=$HF_HOME"
nvidia-smi | grep -E "Driver|GPU 0"

ARGS="--model $HF_MODEL --served-model-name $SERVED_NAME"
ARGS="$ARGS --port $PORT --host 0.0.0.0"
ARGS="$ARGS --max-model-len $MAX_LEN --gpu-memory-utilization $GPU_MEM"
ARGS="$ARGS --download-dir $SCRATCH"
[ "$PREFIX_CACHE" = "true" ] && ARGS="$ARGS --enable-prefix-caching"
[ -n "$EXTRA_ARGS" ] && ARGS="$ARGS $EXTRA_ARGS"

~/.vllm-env/bin/python -m vllm.entrypoints.openai.api_server $ARGS &

VLLM_PID=$!
until curl -sf http://localhost:$PORT/health > /dev/null 2>&1; do sleep 5; done
echo ">>> READY"
echo ">>> Tunnel : ssh -NL $PORT:$(hostname):$PORT <user>@turing.iiit.ac.in"
wait $VLLM_PID
