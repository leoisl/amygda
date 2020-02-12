import random
import os
import numpy as np
import cv2 as cv
import pandas as pd
import argparse

game_items = ['🧲', '🐚', '🦷', '🦴', '🗿', '🍍', '🥀']
items = []
hp = 0
GAME_TITLE = 'Bug Lasso'

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


def get_avg_pixel_intensity_for_contour(image, contour):
    intensities = []
    x_rect, y_rect, w_rect, h_rect = cv.boundingRect(contour)

    for x in range(int(x_rect), int(x_rect+w_rect+1)):
        for y in range(int(y_rect), int(y_rect+h_rect+1)):
            point_in_contour = cv.pointPolygonTest(contour, (x,y), False) == 1.0
            if point_in_contour:
                intensities.append(image[y][x])

    if len(intensities) == 0:
        return 255
    else:
        return np.mean(intensities)

if __name__ == "__main__":
    args = get_args()
    wells_paths = pd.read_csv(args.wells_csv)["wells"]

    cv.namedWindow(GAME_TITLE, cv.WINDOW_GUI_NORMAL)

    cv.createTrackbar('thickness', GAME_TITLE, 3, 100, null_fn)
    cv.createTrackbar('white_noise', GAME_TITLE, 3, 50, null_fn)
    cv.createTrackbar('min growth', GAME_TITLE, 0, 500, null_fn)
    cv.createTrackbar('dilation', GAME_TITLE, 0, 40, null_fn)
    cv.createTrackbar('closing', GAME_TITLE, 0, 20, null_fn)
    # cv.createTrackbar('well shadow', GAME_TITLE, 10, 70, null_fn)
    # cv.createTrackbar('max avg pixel intensity', GAME_TITLE, 0, 255, null_fn)

    # default positions
    cv.setTrackbarPos('thickness', GAME_TITLE, 17)
    cv.setTrackbarPos('white_noise', GAME_TITLE, 6)
    cv.setTrackbarPos('min growth', GAME_TITLE, 28)
    # cv.setTrackbarPos('well shadow', GAME_TITLE, 14)
    # cv.setTrackbarPos('max avg pixel intensity', GAME_TITLE, 120)
    cv.setTrackbarPos('dilation', GAME_TITLE, 20)
    cv.setTrackbarPos('closing', GAME_TITLE, 0)


    calls = ["filepath,thickness,white_noise,area_threshold,growth,nb_of_contours,pass_or_fail"]
    well_no = 0
    while well_no < len(wells_paths):
        well_path = wells_paths[well_no]
        well = cv.imread(well_path)
        flags = set([])

        while True:
            # p4 = cv.getTrackbarPos('well shadow', GAME_TITLE)
            # if p4 < 1:
            #     p4=1
            b = 20
            img_border = cv.copyMakeBorder(well, b, b, b, b, cv.BORDER_CONSTANT, 0)
            well2 = cv.cvtColor(well, cv.COLOR_BGR2GRAY)
            well2 = cv.copyMakeBorder(well2, b, b, b, b, cv.BORDER_CONSTANT, 0)
            # circs = cv.HoughCircles(well2, cv.HOUGH_GRADIENT, 1, 1, param1=20, param2=p4, minRadius=37, maxRadius=42)
            # if circs is not None:
            #     #print(len(circs), len(circs[0]), "inner circles")
            #     for cs in circs:
            #         #print(cs)
            #         for c in cs:
            #             x, y, r = c
            #             r = int(r)
            #             x = int(x)
            #             y = int(y)
            #             #cv.circle(well2, (x, y), r, 255, thickness=1)
            #             mask = np.zeros(well2.shape, np.uint8)
            #             cv.circle(mask, (x, y), r, 255, thickness=-1)
            #             well2 = cv.bitwise_and(well2, well2, mask=mask)

            blnk = well2.copy()
            img_blnk = img_border.copy()
            img_blnk_gray = cv.cvtColor(img_blnk, cv.COLOR_BGR2GRAY)
            thickness = cv.getTrackbarPos('thickness', GAME_TITLE)
            if thickness % 2 != 1:
                thickness += 1
            if thickness < 3:
                thickness = 3
            white_noise = cv.getTrackbarPos('white_noise', GAME_TITLE)
            area_thresh = cv.getTrackbarPos('min growth', GAME_TITLE)
            dilation = cv.getTrackbarPos('dilation', GAME_TITLE) - 20
            closing = cv.getTrackbarPos('closing', GAME_TITLE)

            # max_avg_pixel_intensity = cv.getTrackbarPos('max avg pixel intensity', GAME_TITLE)
            blnk = cv.adaptiveThreshold(blnk, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, thickness, white_noise)

            if dilation > 0:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (dilation, dilation))
                blnk = cv.dilate(blnk, kernel, 1)
            elif dilation < 0:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (-dilation, -dilation))
                blnk = cv.erode(blnk, kernel, 1)

            if closing > 0:
                kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (closing, closing))
                blnk = cv.morphologyEx(blnk, cv.MORPH_CLOSE, kernel)

            mask = np.zeros(well2.shape, np.uint8)

            contours = cv.findContours(blnk, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            total_area = 0
            n_contours = 0
            for cnt in contours[0]:
                area = cv.contourArea(cnt)
                contour_has_good_area = area > area_thresh and area < 500
                if contour_has_good_area:
                    # contour_avg_pixel_intensity = get_avg_pixel_intensity_for_contour(img_blnk_gray, cnt)
                    #
                    # contour_has_good_pixel_intensity = contour_avg_pixel_intensity <= max_avg_pixel_intensity
                    # if contour_has_good_pixel_intensity:
                    cv.drawContours(img_blnk, [cnt], -1, (0, 255, 0), 1)
                    total_area += area
                    n_contours += 1

            font = cv.FONT_HERSHEY_SIMPLEX
            img_blnk = cv.putText(img_blnk, str(total_area), (0, img_blnk.shape[1] - 5), font, 0.7, (0, 255, 0), 1)  # , cv.LINE_AA)
            img_blnk = cv.putText(img_blnk, ','.join(flags), (0, 12), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)

            cv.imshow(GAME_TITLE, img_blnk)

            key = cv.waitKey(1)
            if key == 27:
                save_rows(calls, args.output_csv)
                exit()
            elif key == ord('b'):
                if 'BUBBLE' in flags:
                    flags.remove('BUBBLE')
                else:
                    flags.add('BUBBLE')

            elif key == ord('c'):
                if 'COND' in flags:
                    flags.remove('COND')
                else:
                    flags.add('COND')

            elif key == ord('d'):
                if 'DRY' in flags:
                    flags.remove('DRY')
                else:
                    flags.add('DRY')

            elif key == ord('n'):
                flags = ':'.join(flags)
                call = f"{well_path},{thickness},{white_noise},{area_thresh},{total_area},{n_contours},PASS,{flags}"
                calls.append(call)
                well_no += 1
                hp += 1
                status(calls)
                break
            elif key == ord('f'):
                flags = ':'.join(flags)
                call = f"{well_path},{thickness},{white_noise},{area_thresh},{total_area},{n_contours},FAIL,{flags}"
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
