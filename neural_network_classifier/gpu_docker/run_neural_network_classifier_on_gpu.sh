#!/usr/bin/env bash
set -eux

# make sure singularity v3 is being used
module load singularity/3.5.0

# configs that dont change
cuda_bins="/usr/local/cuda/bin"
cuda_libs="/usr/local/cuda/lib64/"

# configs that rarely change
container="leandroishilima_amygda_autokeras_gpu_0.0.1-2020-02-26-394c9c0c3f01.sif"
job_name="amygda_neural_network_gpu"
mem=100000 # ok for hosts with Quadro M6000
num_gpus=1
gpu_opts="num=${num_gpus}:j_exclusive=yes"
gpu_host="gpu-001 gpu-002 gpu-003 gpu-004 gpu-005 gpu-006 gpu-007 gpu-008" # these have Quadro M6000 - we get these exclusively for us
# gpu_host="gpu-009 gpu-010" # these have Tesla V100 (but gpu-011 has issues with container as of now)
num_cpus=40 # ok for hosts with Quadro M6000

# configs that change frequently
max_trials=1000
epochs=1000
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
    -n 1 \
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
        --seed $seed \
        --threads $num_cpus
