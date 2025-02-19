#!/bin/bash
#SBATCH --job-name=rip
#SBATCH --nodes=8
#SBATCH --partition=standard-g
#SBATCH --time=00-02:00:00
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --ntasks-per-node=8

# configure the following

INPUT_FILE=tulu3-en_usable-llama-3.3-70B-generated.jsonl
OUTPUT_FILE=tulu3-en_usable-llama-3.3-70B-scored.jsonl


###
# don't change these
# this model has hardcoded support right now and should not be changed
MODEL=RLHFlow/ArmoRM-Llama3-8B-v0.1

# NOTE:
# BE SURE TO UPDATE SBATCH --ntasks-per-node to work with GPUS_PER_TASK
#
GPUS_PER_TASK=1  # enough for the model and large batch size

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
    python rip.py score \
        --dispatcher_server ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --reward_model_path '"$MODEL"'
'


