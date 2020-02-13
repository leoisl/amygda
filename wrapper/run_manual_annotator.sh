#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <user_id> <wells or extra>"
    echo "Example: $0 Curie wells"
    exit 1
fi

# configs
amygda_home="$HOME/amygda/"
dataset_folder="/media/penelopeprime/bacterial_growth_imaging"
virtualenv_activate_script="$HOME/training/.local/share/virtualenvs/amygda-CYtbES-g/bin/activate"
user=$1
wells_or_extra=$2
output_folder="${dataset_folder}/outputs/${user}/"

source "$virtualenv_activate_script"
cd "$dataset_folder" || exit
mkdir -p "$output_folder"

python "${amygda_home}"/manual_annotator/manual_annotator.py  \
--wells_csv "${dataset_folder}"/participants_datasets/"${user}"/"${wells_or_extra}".csv \
--output_csv "${output_folder}"/output_"${user}"_"${wells_or_extra}".csv

deactivate
