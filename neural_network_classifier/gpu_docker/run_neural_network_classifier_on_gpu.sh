#!/usr/bin/env bash
set -eux

# make sure singularity v3 is being used
module load singularity/3.5.0

# configs that dont change
cuda_bins="/usr/local/cuda/bin"
cuda_libs="/usr/local/cuda/lib64/"
num_gpus=1
gpu_opts="num=${num_gpus}:j_exclusive=yes"
gpu_host="gpu74-v100-hosts"

# configs that rarely change
container="amygda_autokeras_gpu_0.0.1.sif"
job_name="amygda_neural_network_gpu"
mem=21000

# configs that change frequently
max_trials=3
epochs=3
val_split="0.2"
seed=42

bsub -R "select[mem>${mem}] rusage[mem=${mem}]" \
    -M "$mem" \
    -P gpu \
    -gpu "$gpu_opts" \
  	-m "$gpu_host" \
    -o "$job_name".o \
    -e "$job_name".e \
    -J "$job_name" \
    singularity exec \
        --bind "$cuda_bins" --bind "$cuda_libs"  \
        --nv \
        "$container" \
        python get_model.py \
        --training_data /hps/nobackup/research/zi/leandro/amygda/neural_network_classifier/outputs \
        --classifier_name classifier_NN_on_gpu.max_trials_${max_trials}.epochs_${epochs}.val_split_${val_split}.seed_${seed} \
        --max_trials $max_trials \
        --epochs $epochs \
        --val_split $val_split \
        --seed $seed
