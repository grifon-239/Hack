import cv2
import numpy as np


def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def split_image_with_overlap(image, part_width, part_height, overlap):
    img_width, img_height, _ = image.shape

    parts = []
    parts_dict = {}
    ind = 1
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            box = (x, y, min(x + part_width, img_width), min(y + part_height, img_height))
            part = image[box[1]:box[3], box[0]:box[2]]

            parts.append(part)
            parts_dict[ind] = [box[1], box[3], box[0], box[2]]

            ind += 1
            x += part_width - overlap
        y += part_height - overlap

    return parts, parts_dict


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image


def equalize_histogram(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def process_images(image1, brightness=0, contrast=0):
    adjusted_image1 = adjust_brightness_contrast(image1, brightness, contrast)
    equalized_image1 = equalize_histogram(adjusted_image1)

    return equalized_image1


def compare_pics(xfeat, image_crop, image_slice, threshold):
    img_crop = cv2.resize(image_crop, (915, 915), interpolation=cv2.INTER_AREA)
    img_slice = cv2.resize(image_slice, (915, 915), interpolation=cv2.INTER_AREA)

    img_slice = process_images(img_slice, brightness=5, contrast=5)
    img_crop = process_images(img_crop, brightness=5, contrast=5)

    mkpts0, mkpts1 = xfeat.match_xfeat_star(img_crop, img_slice, top_k=8000)

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 10, maxIters=37_000, confidence=0.999)

    mm = np.where(mask > 0)
    mkpts0 = mkpts0[mm[0], :]
    mkpts1 = mkpts1[mm[0], :]

    print('Points total:', len(mkpts0))
    if len(mkpts0) > threshold:
        return mkpts0, mkpts1
    else:
        return None, None


def find_target_slice(pts_slice, parts_coords):
    average_point = np.mean(pts_slice, axis=0)

    y_1 = parts_coords[0]
    x_1 = parts_coords[2]

    # recalculate average point coord since its on 915 shape
    point_slice_x = average_point[0] * 3
    point_slice_y = average_point[1] * 3

    # recalculate average point coord on whole image
    point_whole_x = x_1 + point_slice_x
    point_whole_y = y_1 + point_slice_y

    # calculate slice corrds
    x_1_target = max(int(point_whole_x - 2745 / 2), 0)
    x_2_target = max(int(point_whole_x + 2745 / 2), 2745)

    y_1_target = max(int(point_whole_y - 2745 / 2), 0)
    y_2_target = max(int(point_whole_y + 2745 / 2), 2745)

    target_slice_coords = [y_1_target, y_2_target, x_1_target, x_2_target]

    return target_slice_coords


def find_corners(ref_points, dst_points):
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    corners = []

    h = 915
    w = 915

    corners_img1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))

        # bottom left, upper left, upper right, bottom right
        corners.append(start_point)

    return corners


def get_final_coords(points, target_slice, target_slice_coords):
    point_1 = (points[1][0] * 3, points[1][1] * 3)
    point_2 = (points[3][0] * 3, points[3][1] * 3)

    cv2.rectangle(target_slice, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])), (0, 0, 255), 2)
    cv2.imwrite('static/slice_res.jpg', target_slice)

    final_point_1_x = int(target_slice_coords[2] + point_1[0])
    final_point_1_y = int(target_slice_coords[0] + point_1[1])

    final_point_2_x = int(target_slice_coords[2] + point_2[0])
    final_point_2_y = int(target_slice_coords[0] + point_2[1])

    # top left - 1, bottom right - 2, top right - 3, bottom left - 4
    final_point_1 = (final_point_1_x, final_point_1_y)
    final_point_2 = (final_point_2_x, final_point_1_y)
    final_point_3 = (final_point_2_x, final_point_2_y)
    final_point_4 = (final_point_1_x, final_point_2_y)

    return final_point_1, final_point_2, final_point_3, final_point_4
