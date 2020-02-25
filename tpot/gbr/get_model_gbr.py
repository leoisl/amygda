import pandas as pd
import glob
import argparse
import cv2
import numpy as np
from tpot import TPOTRegressor

regressor_config_dict = {
    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': range(50, 500, 50),
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.25, 0.5, 0.75, 1.],
        'max_depth': range(1, 20),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
        'criterion': ["mae"],
        'validation_fraction': [0.25],
        'n_iter_no_change': [20]
    }
}


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
        image = image.flatten()
        images.append(image)
    images = np.array(images)
    return images

def get_args():
    parser = argparse.ArgumentParser(description='Create a machine learning model for bacterial growth regression.')
    parser.add_argument('--training_data', type=str, help='Path to the labelled training data', required=True)
    parser.add_argument('--classifier_name', type=str, help='A name for this classifier', required=True)
    parser.add_argument('--generations', type=int, help='Number of generations', default=100)
    parser.add_argument('--population_size', type=int, help='Population size', default=100)
    parser.add_argument('--threads', type=int, help='Number of threads', required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    df = get_df_with_all_calls(args.training_data)
    df_pass = df[df.pass_or_fail == "PASS"]
    df_pass = df_pass[df_pass.growth <= 1000]
    df_pass_shuffled = df_pass.sample(frac=1)
    images = load_images(df_pass_shuffled)
    growths = df_pass_shuffled["growth"]

    tpot = TPOTRegressor(generations=args.generations, population_size=args.population_size, verbosity=2,
                         scoring="neg_median_absolute_error", n_jobs=args.threads, max_eval_time_mins=10,
                         periodic_checkpoint_folder=f"{args.classifier_name}_tpot", config_dict=regressor_config_dict)
    tpot.fit(images, growths)
    tpot.export(f'{args.classifier_name}_pipeline.py')


if __name__=="__main__":
    main()
