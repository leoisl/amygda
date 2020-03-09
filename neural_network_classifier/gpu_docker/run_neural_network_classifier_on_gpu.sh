#!/usr/bin/env bash
# set -eux

# make sure singularity v3 is being used
# module load singularity/3.5.0

# configs that rarely change
container="leandroishilima_amygda_autokeras_gpu_0.0.1-2020-02-26-394c9c0c3f01.sif"
mem=15000
num_gpus=1
gpu_opts="num=${num_gpus}:j_exclusive=yes"
gpu_host="gpu-009 gpu-010" # these have Tesla V100 (but gpu-011 has issues with container as of now)

# configs that change frequently
nb_of_jobs=4
max_trials=400
epochs=500
val_split="0.3"


for i in $(seq 1 ${nb_of_jobs})
do
  job_name="amygda_neural_network_gpu_iteration_${i}"
  seed=${i}

  echo bsub -R "select[mem>${mem}] rusage[mem=${mem}]" \
    -M "$mem" \
    -P gpu \
    -gpu "$gpu_opts" \
  	-m "$gpu_host" \
    -o "$job_name".o \
    -e "$job_name".e \
    -J "$job_name" \
    singularity exec \
        --nv \
        "$container" \
        python get_model.py \
        --training_data outputs \
        --classifier_name classifier_NN_on_gpu.max_trials_${max_trials}.epochs_${epochs}.val_split_${val_split}.iteration_${i}.seed_${seed} \
        --max_trials $max_trials \
        --epochs $epochs \
        --val_split $val_split \
        --seed $seed
done


