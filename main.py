import argparse
import os.path
import warnings

from osgeo import gdal
import numpy as np
import torch
import cv2
import pandas as pd
import time

from preprocessing import make_png_from_tiff
from image_utils import split_image_with_overlap, compare_pics, find_target_slice, find_corners, get_final_coords
from geo_utils import pixel_2_cord, create_geo_json, png2Tif
from defect_pixels import find_defect_pixels

warnings.filterwarnings('ignore')

EPSG_SAVE_PATH = 'dataset/'
GEO_JSON_SAVE_PATH = 'dataset/'
DEFECT_PIXELS_SAVE_PATH = 'dataset/'

SLICE_WIDTH = 2745
SLICE_HEIGHT = 2745
OVERLAP = 0
XFEAT_THRESHOLD = 100
SLICE_MATH_FLAG = False
SAVE_IMAGE_CORRECTED_TIF = False
CREATE_GEO_JSON = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_name', type=str, default='dataset/crops/crop_0_0_0000.tif',
                        help='Path to crop file for testing (should be .tif)')
    parser.add_argument('--layout_name', type=str, default='dataset/layouts/layout_2021-08-16.tif',
                        help='Path to reference layout file (should be .tif)')

    parser.add_argument('--path2save_coords', type=str, default='dataset',
                        help='Path to save coords.csv')

    # parse configs
    args = parser.parse_args()

    # make shure both files in .tif format
    crop_path_tif = args.crop_name
    layout_path_tif = args.layout_name

    path2save_coord = args.path2save_coords
    name_layout_tif = os.path.split(layout_path_tif)[1]
    name_crop_tif = os.path.split(crop_path_tif)[1]

    if crop_path_tif.split('.')[-1] != 'tif' or layout_path_tif.split('.')[-1] != 'tif':
        raise FileNotFoundError(f'Incorect file file format: both files should have ".tif" extension!')

    # read gdal files and make png from tif
    crop_tif = gdal.Open(crop_path_tif, gdal.GA_ReadOnly)
    layout_tif = gdal.Open(layout_path_tif, gdal.GA_ReadOnly)

    image_crop = make_png_from_tiff(gdal_file=crop_tif)
    image_layout = make_png_from_tiff(gdal_file=layout_tif)

    # load xfeat

    start_time = time.time()

    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)

    # create slices coords for whole image
    parts, parts_dict = split_image_with_overlap(image_layout, SLICE_WIDTH, SLICE_HEIGHT, OVERLAP)

    # loop through slices and find a target slice that matches with a crop
    for i, test_slice in enumerate(parts):
        i += 1
        crop_points, slice_points = compare_pics(xfeat, image_crop, test_slice, threshold=XFEAT_THRESHOLD)

        if type(crop_points) is np.ndarray:
            SLICE_MATH_FLAG = True
            break

    # calculate and find new target slice with points in its center
    if SLICE_MATH_FLAG:
        target_slice_coords = find_target_slice(slice_points, parts_dict[i])

        # found target slice coords, take whole image and perform calc on new slice
        target_slice = image_layout[target_slice_coords[0]:target_slice_coords[1],
                       target_slice_coords[2]:target_slice_coords[3]]
        target_crop_pts, target_slice_pts = compare_pics(xfeat, image_crop, target_slice, threshold=XFEAT_THRESHOLD)

        # get rectangle coords for matches
        points = find_corners(target_crop_pts, target_slice_pts)

        # get crop coords on layout: top left - 1, bottom right - 2, top right - 3, bottom left - 4
        fp_1, fp_2, fp_3, fp_4 = get_final_coords(points, target_slice, target_slice_coords)

        # get geo info from layout tif and make found coords in EPSG and write to file
        geotransform = layout_tif.GetGeoTransform()
        points_EPSG = pixel_2_cord([fp_1, fp_2, fp_3, fp_4], geotransform, EPSG_SAVE_PATH)

        end_time = time.time()

        df = pd.DataFrame({'layout_name': [name_layout_tif],
                           'crop_name': [name_crop_tif],
                           'ul': [(points_EPSG[0][0], points_EPSG[0][1])],
                           'ur': [(points_EPSG[1][0], points_EPSG[1][1])],
                           'br': [(points_EPSG[2][0], points_EPSG[2][1])],
                           'bl': [(points_EPSG[3][0], points_EPSG[3][1])],
                           'crs': ['EPSG:32637'],
                           'start': [start_time],
                           'end': [end_time],
                           'elapsed_time': [end_time - start_time]
                           })

        df.to_csv(path2save_coord, index=False)

        # create geojson coords and write to file

        if CREATE_GEO_JSON:
            create_geo_json(points_EPSG, GEO_JSON_SAVE_PATH)
        else:
            pass

        # second task: find defective pixels: return crops with corrected pixels and save a corrections report
        crop_image_corrected = find_defect_pixels(crop_image=image_crop, save_path=DEFECT_PIXELS_SAVE_PATH)
        cv2.imwrite(DEFECT_PIXELS_SAVE_PATH + 'crop_corrected.jpg', crop_image_corrected)

        if SAVE_IMAGE_CORRECTED_TIF:
            png2Tif(os.path.join(DEFECT_PIXELS_SAVE_PATH, 'crop_corrected.jpg'), DEFECT_PIXELS_SAVE_PATH, points_EPSG)
        else:
            pass

        print(f'Processing done: EPSG crop coordinates: {points_EPSG[0]}, {points_EPSG[1]},'
              f' {points_EPSG[2]}, {points_EPSG[3]}')

    else:
        print('Given crop not found on a layout!')
