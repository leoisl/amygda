import pandas as pd
import argparse
import os
import subprocess
def run_command(command, dry_run):
    print(f"Running {command}...")
    if not dry_run:
        subprocess.check_call(command, shell=True)


def get_args():
    parser = argparse.ArgumentParser(description='Create folder with new images to classify')
    parser.add_argument('--input_csv', type=str, help='Path to input csv. Must have PATH column.', required=True)
    parser.add_argument('--folder_with_new_images', type=str, help='Path to the folder with new images', required=True)
    parser.add_argument('--dry-run', help='Do not perform the copy, just print the commands.', default=False,
                        action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    input_csv = args.input_csv
    folder_with_new_images = args.folder_with_new_images
    dry_run = args.dry_run
    os.makedirs(folder_with_new_images, exist_ok=True)
    df = pd.read_csv(input_csv)
    df = df.dropna()
    for path in df.PATH:
        run_command(f"cp {path} {folder_with_new_images}", dry_run)


if __name__=="__main__":
    main()