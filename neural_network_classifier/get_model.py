import pandas as pd
import glob
import argparse
import cv2
import numpy as np
import autokeras as ak


def get_df_with_all_calls(path_to_outputs):
    all_csvs = glob.glob(f'{path_to_outputs}/*/*.csv')
    all_csvs = [csv for csv in all_csvs if not csv.endswith("training.csv")]
    all_dfs = [pd.read_csv(csv) for csv in all_csvs]
    df = pd.concat(all_dfs, ignore_index=True)
    return df

def load_images(df_pass_shuffled):
    images = []
    for well_path in df_pass_shuffled["well_path"]:
        image = cv2.imread(well_path, cv2.IMREAD_GRAYSCALE)
        image = np.where(image==0, 255, image)
        image = cv2.resize(image, (82, 82))
        images.append(image)
    images = np.array(images)
    return images

def get_args():
    parser = argparse.ArgumentParser(description='Create a neural network model for bacterial growth classification.')
    parser.add_argument('--training_data', type=str, help='Path to the labelled training data', required=True)
    parser.add_argument('--classifier_name', type=str, help='A name for this classifier', required=True)
    parser.add_argument('--max_trials', type=int, help='Number of different Keras models to try', default=100)
    parser.add_argument('--epochs', type=int, help='Number of epochs for the fit method', default=1000)
    parser.add_argument('--threads', type=int, help='Number of threads', required=True)
    parser.add_argument('--val_split', type=float, help='Validation split', required=True)
    parser.add_argument('--resume', action="store_true", help='Resume previous training', required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    df = get_df_with_all_calls(args.training_data)
    df_pass = df[df.pass_or_fail == "PASS"]
    df_pass = df_pass[df_pass.growth <= 1000]
    # df_pass = df_pass[df_pass.growth >= 20]
    df_pass_shuffled = df_pass.sample(frac=1)
    images = load_images(df_pass_shuffled)
    growths = df_pass_shuffled["growth"]

    # create regressor
    clf = ak.ImageRegressor(max_trials=args.max_trials, name=args.classifier_name,
                            loss="mae", metrics=['mse', 'mae', 'mape'], objective="val_mae", overwrite=not args.resume)

    # train
    clf.fit(images, growths, epochs=args.epochs, workers=args.threads, validation_split=args.val_split)

    # save to disk
    model = clf.export_model()
    model.save(f"{args.classifier_name}.h5")

if __name__=="__main__":
    main()
