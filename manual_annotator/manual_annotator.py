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

def show_images(human_vision, computer_vision):
    image_to_show = np.hstack((human_vision, computer_vision))
    cv.imshow(GAME_TITLE, image_to_show)


def mask_circles(well_gray_with_border, p4):
    circs = cv.HoughCircles(well_gray_with_border, cv.HOUGH_GRADIENT, 1, 1, param1=20, param2=p4, minRadius=37,
                            maxRadius=42)
    if circs is not None:
        for cs in circs:
            for c in cs:
                x, y, r = c
                r = int(r)
                x = int(x)
                y = int(y)
                mask = np.zeros(well_gray_with_border.shape, np.uint8)
                cv.circle(mask, (x, y), r, 255, thickness=-1)
                well_gray_with_border = cv.bitwise_and(well_gray_with_border, well_gray_with_border, mask=mask)
    return well_gray_with_border

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
            # get input
            p4 = cv.getTrackbarPos('well shadow', GAME_TITLE)
            if p4 < 1:
                p4=1
            b = 40
            contour_thickness = cv.getTrackbarPos('contour_thickness', GAME_TITLE)
            if contour_thickness % 2 != 1:
                contour_thickness += 1
            if contour_thickness < 3:
                contour_thickness = 3
            white_noise_remover = cv.getTrackbarPos('white_noise_remover', GAME_TITLE)
            min_area_thresh = cv.getTrackbarPos('min growth', GAME_TITLE)
            max_area_thresh = cv.getTrackbarPos('max growth', GAME_TITLE)



            well_gray_with_border = cv.cvtColor(well, cv.COLOR_BGR2GRAY)
            well_gray_with_border = cv.copyMakeBorder(well_gray_with_border, b, b, b, b, cv.BORDER_CONSTANT, 0)
            well_gray_with_border = mask_circles(well_gray_with_border, p4)
            binarized_image = well_gray_with_border.copy()
            binarized_image = cv.adaptiveThreshold(binarized_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, contour_thickness, white_noise_remover)

            well_with_border = cv.copyMakeBorder(well, b, b, b, b, cv.BORDER_CONSTANT, 0)
            binarized_image_with_color = cv.cvtColor(binarized_image, cv.COLOR_GRAY2BGR)


            # find and draw contours
            contours = cv.findContours(binarized_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[0]
            total_area = 0
            n_contours = 0
            color_index = 0

            good_contours = []
            for cnt in contours:
                area = cv.contourArea(cnt)
                contour_has_good_area = area >= min_area_thresh and area <= max_area_thresh
                if contour_has_good_area and not is_a_forbidden_contours(cnt):
                    color_index, color = get_color(color_index)
                    cv.drawContours(well_with_border, [cnt], -1, color, 1)
                    cv.drawContours(binarized_image_with_color, [cnt], -1, color, 1)
                    total_area += area
                    n_contours += 1
                    good_contours.append(cnt)


            # write info to images
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(well_with_border, f"Growth: {total_area}", (0, well_with_border.shape[1] - 5), font, 0.7, (0, 255, 0), 1)  # , cv.LINE_AA)
            cv.putText(well_with_border, ','.join(flags), (0, 12), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)
            if len(forbidden_contours):
                cv.putText(well_with_border, f"{len(forbidden_contours)} REMOVED", (0, 28), font, 0.5, (0, 0, 255), 1)  # , cv.LINE_AA)

            # hotkeys_image = np.full(shape=(70, well_with_border.shape[1]+binarized_image_with_color.shape[1]),
            #                         fill_value=np.uint8(255))
            # hotkeys_image = cv.cvtColor(hotkeys_image, cv.COLOR_GRAY2BGR)
            # hotkeys_image = cv.putText(hotkeys_image, "[Enter]=Confirm; [F]=Fail; [P]=Previous;", (1, 20), font, 0.5, (0, 0, 0), 1)
            # hotkeys_image = cv.putText(hotkeys_image, "[B]=Bubble; [C]=Cond; [D]=Dry well;", (1, 40), font, 0.5, (0, 0, 0), 1)
            # hotkeys_image = cv.putText(hotkeys_image, "[ESC]=Save and quit;", (1, 60), font, 0.5, (0, 0, 0), 1)

            show_images(well_with_border, binarized_image_with_color)


            key = cv.waitKey(25)
            if key == 27: # esc
                save_rows(calls_csv, args.output_csv)
                exit()
            elif key == ord('b') or key == ord('B'):
                if 'BUBBLE' in flags:
                    flags.remove('BUBBLE')
                else:
                    flags.add('BUBBLE')
            elif key == ord('c') or key == ord('C'):
                if 'COND' in flags:
                    flags.remove('COND')
                else:
                    flags.add('COND')
            elif key == ord('d') or key == ord('D'):
                if 'DRY' in flags:
                    flags.remove('DRY')
                else:
                    flags.add('DRY')
            elif key == 13: # enter
                flags = ':'.join(flags)
                calls_csv.loc[well_path] = [contour_thickness, white_noise_remover, min_area_thresh, total_area, n_contours, "PASS", flags]
                well_no += 1
                break
            elif key == ord('f') or key == ord('F'):
                flags = ':'.join(flags)
                calls_csv.loc[well_path] = [contour_thickness, white_noise_remover, min_area_thresh, total_area, n_contours, "FAIL", flags]
                well_no += 1
                break
            elif key == ord('p') or key == ord('P'):
                if well_no > 0: well_no -= 1
                break


    save_rows(calls_csv, args.output_csv)
    cv.destroyAllWindows()
