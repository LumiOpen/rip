#!/bin/bash
#SBATCH --job-name=rip
#SBATCH --nodes=64
#SBATCH --partition=standard-g
#SBATCH --time=00-48:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


# configure the following

# Input file should be jsonl in the format:
# {..., "messages": [{"role": "assistant", # "content": "..."}, ...}
# The value in messages[0].content will be used as the prompt.

INPUT_FILE=/scratch/project_462000353/jburdge/data/tulu-3/en_usable.jsonl

OUTPUT_FILE=tulu3-en_usable-llama-3.3-70B-generated.jsonl

MODEL=meta-llama/Llama-3.3-70B-Instruct
GPUS_PER_TASK=4  # enough for the model and large batch size
#
# BE SURE TO UPDATE SBATCH --ntasks-per-node to work with GPUS_PER_TASK
#

###


export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999

mkdir -p logs output

module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_462000353/jburdge/venv/bin/activate

python -m dispatcher.server \
    --infile $INPUT_FILE \
    --outfile $OUTPUT_FILE \
    --host 0.0.0.0 \
    --port ${DISPATCHER_PORT} &

sleep 10

srun -l bash -c '
    # Compute the starting GPU index for this task.
    # SLURM_LOCALID is the index of the task on this node.
    start_gpu=$(( SLURM_LOCALID * '"$GPUS_PER_TASK"' ))
    GPU_IDS=""
    for (( i=0; i < '"$GPUS_PER_TASK"'; i++ )); do
        if [ -z "$GPU_IDS" ]; then
            GPU_IDS="$(( start_gpu + i ))"
        else
            GPU_IDS="${GPU_IDS},$(( start_gpu + i ))"
        fi
    done
    export CUDA_VISIBLE_DEVICES=$GPU_IDS

	# Set ports uniquely per task (to avoid collisions)
    export MASTER_PORT=$(( 7000 + SLURM_LOCALID ))
    export VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch
    python rip.py generate \
        --dispatcher_server ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --max_model_len 16384 \
        --tensor_parallel_size '"$GPUS_PER_TASK"' \
        --base_model_path '"$MODEL"'
'

