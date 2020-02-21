import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='Load a neural network model for bacterial growth classification and predicts bacterial growth.')
    parser.add_argument('--model', type=str, help='Path to the neural network model', required=True)
    parser.add_argument('--images', type=str, help='Text file with images to predict', required=True)
    parser.add_argument('--threads', type=int, help='Number of threads', required=True)
    parser.add_argument('--output_csv', type=str, help='Prediction output csv', required=True)
    args = parser.parse_args()
    return args

def load_images(images_filepaths):
    images = []
    for image_filepath in images_filepaths:
        image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (82, 82))
        image = img_to_array(image)
        images.append(image)
    images = np.array(images)
    return images


def main():
    args = get_args()
    model = load_model(args.model)
    print("Loaded model:")
    print(model.summary())

    with open(args.images) as images_fh:
        images_filenames = images_fh.readlines()
    images_filenames = [image_filename.strip() for image_filename in images_filenames if len(image_filename.strip()) > 0]
    images = load_images(images_filenames)

    predictions = model.predict(images, workers=args.threads)
    predictions_as_list = list(predictions.flatten())
    df = pd.DataFrame({"well_path": images_filenames, "predicted_growth": predictions_as_list})
    df.to_csv(args.output_csv, index=False)

if __name__=="__main__":
    main()
