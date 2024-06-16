import os

import cv2
import numpy as np


def find_defect_pixels(crop_image, save_path):
    low_border = 0.15
    high_border = 5

    with open(os.path.join(save_path, 'pixels_result.txt'), 'w') as wr:
        # [номер строки]; [номер столбца]; [номер канала]; [«битое» значение]; [исправленное значение]

        img_draw = crop_image.copy()
        img_corrected = crop_image.copy()
        h, w, col = crop_image.shape
        dots = []

        for x in range(w):
            for y in range(h):
                for c in range(col):
                    px_val = crop_image[y, x, c]
                    ind_pixels_near_x = [x - 1, x, x + 1]
                    ind_pixels_near_y = [y - 1, y, y + 1]
                    if x == 0:
                        ind_pixels_near_x = [x, x + 1]
                    if y == 0:
                        ind_pixels_near_y = [y, y + 1]
                    if x + 1 == w:
                        ind_pixels_near_x = [x - 1, x]
                    if y + 1 == h:
                        ind_pixels_near_y = [y - 1, y]
                    pixels_near = []
                    for i, x_v in enumerate(ind_pixels_near_x):
                        for j, y_v in enumerate(ind_pixels_near_y):
                            if i == 1 and j == 1 and len(ind_pixels_near_y) == 3 and len(ind_pixels_near_x) == 3:
                                continue
                            pixels_near.append(crop_image[y_v, x_v, c])
                    near_mean_px = np.median([pixels_near])

                    if px_val <= low_border * near_mean_px or px_val == 0 or px_val >= high_border * near_mean_px:
                        img_corrected[y, x, c] = near_mean_px
                        px_val_corrected = near_mean_px
                        dots.append((x, y))
                        line = f"{x}; {y}; {c}; {px_val}; {px_val_corrected}\n"
                        wr.write(line)

    return img_corrected

    # radius = 2
    # color = (255, 255, 0)
    # thickness = 2
    # for (x, y) in dots:
    #     center_coordinates = (x, y)
    #     img_draw = cv2.circle(img_draw, center_coordinates, radius, color, thickness)
    # img = cv2.resize(crop_image, (640, 640))
    # img_corrected = cv2.resize(img_corrected, (640, 640))
    # img_draw = cv2.resize(img_draw, (640, 640))
    # cv2.imshow('img_corrected', img_corrected)
    # cv2.imshow('dots', img_draw)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
