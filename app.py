from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_dropzone import Dropzone
import os, shutil
import os.path
import warnings

from osgeo import gdal
import numpy as np
import torch
import cv2
import pandas as pd
import time
import shutil

from preprocessing import make_png_from_tiff, make_4channels_from_tiff
from image_utils import split_image_with_overlap, compare_pics, find_target_slice, find_corners, get_final_coords
from geo_utils import pixel_2_cord, create_geo_json, png2Tif
from defect_pixels import find_defect_pixels
from image_utils import adjust_gamma

warnings.filterwarnings('ignore')

EPSG_SAVE_PATH = 'result/'
GEO_JSON_SAVE_PATH = 'result/'
DEFECT_PIXELS_SAVE_PATH = 'result/'


SLICE_WIDTH = 2745
SLICE_HEIGHT = 2745
OVERLAP = 0
XFEAT_THRESHOLD = 100
SLICE_MATH_FLAG = False
SAVE_IMAGE_CORRECTED_TIF = True
CREATE_GEO_JSON = True


app = Flask(__name__)
dropzone = Dropzone(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAPS_FOLDER'] = 'layouts'
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_MAX_FILES'] = 100
app.config['DROPZONE_PARALLEL_UPLOADS'] = 100

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MAPS_FOLDER'], exist_ok=True)
os.makedirs(EPSG_SAVE_PATH, exist_ok=True)
os.makedirs(GEO_JSON_SAVE_PATH, exist_ok=True)
os.makedirs(DEFECT_PIXELS_SAVE_PATH, exist_ok=True)
os.makedirs('static', exist_ok=True)


if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    maps = os.listdir(app.config['MAPS_FOLDER'])
    return render_template('index.html', files=files, maps=maps)



@app.route('/upload', methods=['POST'])
def upload():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except:
            pass

    for key, file in request.files.items():
        if key.startswith('file'):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    return '', 204


@app.route('/get_file_path', methods=['POST'])
def get_file_path():
    selected_file = request.form.get('selected_file')
    map_path = os.path.join(app.config['MAPS_FOLDER'], selected_file)

    crop_path_tif = os.path.join('uploads', os.listdir('uploads')[0])
    layout_path_tif = map_path

    path2save_coord = 'result/'
    name_layout_tif = os.path.split(layout_path_tif)[1]
    name_crop_tif = os.path.split(crop_path_tif)[1]

    if crop_path_tif.split('.')[-1] != 'tif' or layout_path_tif.split('.')[-1] != 'tif':
        raise FileNotFoundError(f'Incorect file file format: both files should have ".tif" extension!')

    # read gdal files and make png from tif
    crop_tif = gdal.Open(crop_path_tif, gdal.GA_ReadOnly)
    layout_tif = gdal.Open(layout_path_tif, gdal.GA_ReadOnly)

    image_crop = make_png_from_tiff(gdal_file=crop_tif)
    # cv2.imwrite('static/cropRGB.jpg', image_crop)

    image_crop_4ch = make_4channels_from_tiff(gdal_file=crop_tif)

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

        df.to_csv(os.path.join(path2save_coord, 'coords.csv'), index=False)


        if CREATE_GEO_JSON:
            create_geo_json(points_EPSG, GEO_JSON_SAVE_PATH)
        else:
            pass


        crop_image_corrected = find_defect_pixels(crop_image=image_crop_4ch, save_path=DEFECT_PIXELS_SAVE_PATH)

        crop_image_corrected_2 = crop_image_corrected[:,:,:3]

        crop_image_corrected_2 = adjust_gamma(crop_image_corrected_2, gamma=2.0)

        cv2.imwrite(DEFECT_PIXELS_SAVE_PATH + 'crop_corrected.png', crop_image_corrected_2)
        cv2.imwrite(os.path.join('static', 'crop_corrected.png'), crop_image_corrected_2)
        cv2.imwrite(DEFECT_PIXELS_SAVE_PATH + 'crop_corrected_tmp.tif', crop_image_corrected)

        if SAVE_IMAGE_CORRECTED_TIF:
            png2Tif(os.path.join(DEFECT_PIXELS_SAVE_PATH, 'crop_corrected_tmp.tif'), DEFECT_PIXELS_SAVE_PATH, points_EPSG)
            os.remove(os.path.join(DEFECT_PIXELS_SAVE_PATH, 'crop_corrected_tmp.tif'))
        else:
            pass

        print(f'Processing done: EPSG crop coordinates: {points_EPSG[0]}, {points_EPSG[1]},'
              f'{points_EPSG[2]}, {points_EPSG[3]}')

        resulted_text = f'Расчет завершен, координаты в формате EPSG:32637: {points_EPSG[0]}, {points_EPSG[1]}, {points_EPSG[2]}, {points_EPSG[3]}'


        resulted_image_path = 'static/slice_res.jpg'

        resulted_image_path_2 = os.path.join('static', 'crop_corrected.png')


    else:

        crop_image_corrected = find_defect_pixels(crop_image=image_crop_4ch, save_path=DEFECT_PIXELS_SAVE_PATH)
        cv2.imwrite(os.path.join(DEFECT_PIXELS_SAVE_PATH, 'crop_corrected_tmp.tif'), crop_image_corrected)
        png2Tif(os.path.join(DEFECT_PIXELS_SAVE_PATH, 'crop_corrected_tmp.tif'), DEFECT_PIXELS_SAVE_PATH)
        os.remove(os.path.join(DEFECT_PIXELS_SAVE_PATH, 'crop_corrected_tmp.tif'))
        print('Кроп не нашелся на подложке')
        resulted_text = f'Кроп не нашелся на подложке'
        resulted_image_path = ''
        resulted_image_path_2 = os.path.join('static', 'crop_corrected.png')


    files = os.listdir(app.config['UPLOAD_FOLDER'])
    maps = os.listdir(app.config['MAPS_FOLDER'])
    return render_template('index.html', resulted_text=resulted_text, files=files, maps=maps, resulted_image_path=resulted_image_path, resulted_image_path_2=resulted_image_path_2)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)