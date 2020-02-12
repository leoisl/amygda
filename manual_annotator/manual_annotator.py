import numpy as np
import cv2 as cv
import pandas as pd
import argparse

GAME_TITLE = 'Bug Lasso'

colors=[(255,0,0), (0,255,0), (0,0,255),
            (255,255,0), (255,0,255),
            (0,255,255)]

# TODO: control multithreaded access?
good_contours=[]
forbidden_contours=[]


def get_color(color_index):
    if color_index == len(colors):
        color_index = 0
    color = colors[color_index]
    color_index+=1
    return color_index, color


def save_rows(calls_csv, out_file):
    calls_csv.to_csv(out_file)


def null_fn(x):
    pass

def mouse_callback(event, x, y, flags, params):
    global good_contours, forbidden_contours
    if event==1:
        for cnt in good_contours:
            if cv.pointPolygonTest(cnt, (x,y), False) >= 0:
                if not is_a_forbidden_contours(cnt):
                    forbidden_contours.append(cnt)
    elif event==2:
        forbidden_contours = []


def is_a_forbidden_contours(cnt):
    global forbidden_contours
    return any(np.array_equal(cnt, x) for x in forbidden_contours)


def get_args():
    parser = argparse.ArgumentParser(description='Manual annotator on a list of well images.')
    parser.add_argument('--wells_csv', type=str, help='CSV with the wells, output by create_participants_dataset.py script', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV', required=True)
    args = parser.parse_args()
    return args

def window_is_closed():
    return cv.getWindowProperty(GAME_TITLE, cv.WND_PROP_VISIBLE) == 0.0

def maximize_window():
    cv.setWindowProperty(GAME_TITLE, cv.WND_PROP_FULLSCREEN, 1.0)

if __name__ == "__main__":
    args = get_args()
    wells_paths = pd.read_csv(args.wells_csv)["wells"]

    cv.namedWindow(GAME_TITLE, cv.WINDOW_GUI_NORMAL)
    # maximize_window()
    cv.setMouseCallback(GAME_TITLE, mouse_callback)

    cv.createTrackbar('contour_thickness', GAME_TITLE, 3, 100, null_fn)
    cv.createTrackbar('white_noise_remover', GAME_TITLE, 3, 50, null_fn)
    cv.createTrackbar('min growth', GAME_TITLE, 0, 500, null_fn)
    cv.createTrackbar('max growth', GAME_TITLE, 0, 5000, null_fn)
    cv.createTrackbar('well shadow', GAME_TITLE, 10, 70, null_fn)

    # default positions
    cv.setTrackbarPos('contour_thickness', GAME_TITLE, 17)
    cv.setTrackbarPos('white_noise_remover', GAME_TITLE, 6)
    cv.setTrackbarPos('min growth', GAME_TITLE, 28)
    cv.setTrackbarPos('max growth', GAME_TITLE, 1000)
    cv.setTrackbarPos('well shadow', GAME_TITLE, 14)


    calls_csv = pd.DataFrame(
        columns=["contour_thickness","white_noise_remover","area_threshold","growth","nb_of_contours","pass_or_fail", "flags"])
    calls_csv.index.name = "well_path"
    well_no = 0
    while well_no < len(wells_paths) and not window_is_closed():
        print(calls_csv)
        well_path = wells_paths[well_no]
        well = cv.imread(well_path)
        flags = set([])
        forbidden_contours = []

        while True and not window_is_closed():
            p4 = cv.getTrackbarPos('well shadow', GAME_TITLE)
            if p4 < 1:
                p4=1
            b = 40
            img_border = cv.copyMakeBorder(well, b, b, b, b, cv.BORDER_CONSTANT, 0)
            well2 = cv.cvtColor(well, cv.COLOR_BGR2GRAY)
            well2 = cv.copyMakeBorder(well2, b, b, b, b, cv.BORDER_CONSTANT, 0)
            circs = cv.HoughCircles(well2, cv.HOUGH_GRADIENT, 1, 1, param1=20, param2=p4, minRadius=37, maxRadius=42)
            if circs is not None:
                for cs in circs:
                    for c in cs:
                        x, y, r = c
                        r = int(r)
                        x = int(x)
                        y = int(y)
                        mask = np.zeros(well2.shape, np.uint8)
                        cv.circle(mask, (x, y), r, 255, thickness=-1)
                        well2 = cv.bitwise_and(well2, well2, mask=mask)

            blnk = well2.copy()
            img_blnk = img_border.copy()
            img_blnk_gray = cv.cvtColor(img_blnk, cv.COLOR_BGR2GRAY)
            contour_thickness = cv.getTrackbarPos('contour_thickness', GAME_TITLE)
            if contour_thickness % 2 != 1:
                contour_thickness += 1
            if contour_thickness < 3:
                contour_thickness = 3
            white_noise_remover = cv.getTrackbarPos('white_noise_remover', GAME_TITLE)
            min_area_thresh = cv.getTrackbarPos('min growth', GAME_TITLE)
            max_area_thresh = cv.getTrackbarPos('max growth', GAME_TITLE)

            blnk = cv.adaptiveThreshold(blnk, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, contour_thickness, white_noise_remover)
            mask = np.zeros(well2.shape, np.uint8)

            contours = cv.findContours(blnk, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
            total_area = 0
            n_contours = 0
            color_index = 0

            good_contours = []
            for cnt in contours:
                area = cv.contourArea(cnt)
                contour_has_good_area = area >= min_area_thresh and area <= max_area_thresh
                if contour_has_good_area and not is_a_forbidden_contours(cnt):
                    color_index, color = get_color(color_index)
                    cv.drawContours(img_blnk, [cnt], -1, color, 1)
                    total_area += area
                    n_contours += 1
                    good_contours.append(cnt)

            font = cv.FONT_HERSHEY_SIMPLEX
            img_blnk = cv.putText(img_blnk, f"Growth: {total_area}", (0, img_blnk.shape[1] - 5), font, 0.7, (0, 255, 0), 1)  # , cv.LINE_AA)
            img_blnk = cv.putText(img_blnk, ','.join(flags), (0, 12), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)
            if len(forbidden_contours):
                img_blnk = cv.putText(img_blnk, f"{len(forbidden_contours)} REMOVED", (0, 28), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)

            cv.imshow(GAME_TITLE, img_blnk)


            key = cv.waitKey(100)
            if key == 27:
                save_rows(calls, args.output_csv)
                save_rows(calls_csv, args.output_csv)
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
                calls_csv.loc[well_path] = [contour_thickness, white_noise_remover, min_area_thresh, total_area, n_contours, "PASS", flags]
                well_no += 1
                break
            elif key == ord('f'):
                flags = ':'.join(flags)
                calls_csv.loc[well_path] = [contour_thickness, white_noise_remover, min_area_thresh, total_area, n_contours, "FAIL", flags]
                well_no += 1
                break
            elif key == ord('p'):
                calls = calls[:-1]
                well_no -= 1
                break

    save_rows(calls_csv, args.output_csv)
    cv.destroyAllWindows()
