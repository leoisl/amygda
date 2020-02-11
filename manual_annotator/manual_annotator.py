import random
import os
import numpy as np
import cv2 as cv
import pandas as pd
import argparse

game_items = ['🧲', '🐚', '🦷', '🦴', '🗿', '🍍', '🥀']
items = []
hp = 0
GAME_TITLE = 'Consumption Crush'

def status(calls):
    os.system('clear')
    for line in calls:
        print(line)
    print("Items: ", '  '.join(items))
    print(f"Score: {hp}")
    if random.randint(0,20) == 15:
        print(random.choice(["wow!!", "almost there!!!", "damn! you're good"]))


def save_rows(calls, out_file):
    with open(out_file, 'w') as out_fd:
        for c in calls:
            print(c, file=out_fd)
    print(f"\nSaved progress to {out_file}")


def null_fn(x):
    pass


def get_args():
    parser = argparse.ArgumentParser(description='Manual annotator on a list of well images.')
    parser.add_argument('--wells_csv', type=str, help='CSV with the wells, output by create_participants_dataset.py script', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    wells_paths = pd.read_csv(args.wells_csv)["wells"]

    cv.namedWindow(GAME_TITLE)

    cv.createTrackbar('p1', GAME_TITLE, 3, 20, null_fn)
    cv.createTrackbar('p2', GAME_TITLE, 3, 20, null_fn)
    cv.createTrackbar('p3', GAME_TITLE, 0, 500, null_fn)
    cv.createTrackbar('p4', GAME_TITLE, 10, 70, null_fn)
    cv.setTrackbarPos('p1', GAME_TITLE, 17)
    cv.setTrackbarPos('p2', GAME_TITLE, 6)
    cv.setTrackbarPos('p3', GAME_TITLE, 28)
    cv.setTrackbarPos('p4', GAME_TITLE, 14)
   
    calls = ["filepath,p1,p2,area_threshold,growth,nb_of_contours,pass_or_fail"]
    well_no = 0
    while well_no < len(wells_paths):
        well_path = wells_paths[well_no]
        well = cv.imread(well_path)

        while True:
            p4 = cv.getTrackbarPos('p4', GAME_TITLE)
            if p4 < 1:
                p4=1
            b = 20
            img_border = cv.copyMakeBorder(well, b, b, b, b, cv.BORDER_CONSTANT, 0)
            well2 = cv.cvtColor(well, cv.COLOR_BGR2GRAY)
            well2 = cv.copyMakeBorder(well2, b, b, b, b, cv.BORDER_CONSTANT, 0)
            circs = cv.HoughCircles(well2, cv.HOUGH_GRADIENT, 1, 1, param1=20, param2=p4, minRadius=37, maxRadius=42)
            if circs is not None:
                #print(len(circs), len(circs[0]), "inner circles")
                for cs in circs:
                    #print(cs)
                    for c in cs:
                        x, y, r = c
                        r = int(r)
                        x = int(x)
                        y = int(y)
                        #cv.circle(well2, (x, y), r, 255, thickness=1)
                        mask = np.zeros(well2.shape, np.uint8)
                        cv.circle(mask, (x, y), r, 255, thickness=-1)
                        well2 = cv.bitwise_and(well2, well2, mask=mask)

            blnk = well2.copy()
            img_blnk = img_border.copy()
            p1 = cv.getTrackbarPos('p1', GAME_TITLE)
            if p1 % 2 != 1:
                p1 += 1
            if p1 < 3:
                p1 = 3
            p2 = cv.getTrackbarPos('p2', GAME_TITLE)
            area_thresh = cv.getTrackbarPos('p3', GAME_TITLE)
            blnk = cv.adaptiveThreshold(blnk, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, p1, p2)
            mask = np.zeros(well2.shape, np.uint8)

            contours = cv.findContours(blnk, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            total_area = 0
            n_contours = 0
            for c in contours[0]:
                area = cv.contourArea(c)
                if area > area_thresh and area < 500:
                    cv.drawContours(img_blnk, [c], -1, (0, 255, 0), 1)
                    total_area += area
                    n_contours += 1
                   #break
            cv.imshow(GAME_TITLE, img_blnk)

            key = cv.waitKey(1)
            if key == 27:
                save_rows(calls)
                exit()
            elif key == ord('n'):
                call = f"{well_path},{p1},{p2},{area_thresh},{total_area},{n_contours},PASS"
                calls.append(call)
                well_no += 1
                hp += 1
                status(calls)
                break
            elif key == ord('f'):
                call = f"{well_path},{p1},{p2},{area_thresh},{total_area},{n_contours},FAIL"
                calls.append(call)
                well_no += 1
                hp += 1
                status(calls)
                break
            elif key == ord('p'):
                calls = calls[:-1]
                well_no -= 1
                hp -= 1
                status(calls)
                break

    save_rows(calls, args.output_csv)
    cv.destroyAllWindows()
