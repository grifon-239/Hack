import os

import geojson
import rasterio
from geojson import Feature, Point, FeatureCollection


def pixel_2_cord(points, geotransform, save_path, name_crop_tif):
    pointsCoord = []

    for point in points:
        X_geo = geotransform[0] + point[0] * geotransform[1] + point[1] * geotransform[2]
        Y_geo = geotransform[3] + point[0] * geotransform[4] + point[1] * geotransform[5]

        pointsCoord.append((X_geo, Y_geo))

    with open(os.path.join(save_path, f'EPSG_result_{name_crop_tif.split(".")[0]}.txt'), "w") as f:
        for point in pointsCoord:
            f.write(f'{str(point[0])}; {str(point[1])}\n')

    return pointsCoord[0], pointsCoord[1], pointsCoord[2], pointsCoord[3]


def create_geo_json(pointsCoord: list, path2saveGeoJSON: str, name_crop_tif) -> None:
    upper_left = Feature(geometry=Point((pointsCoord[0][0], pointsCoord[0][1])))
    upper_right = Feature(geometry=Point((pointsCoord[1][0], pointsCoord[1][1])))
    lower_right = Feature(geometry=Point((pointsCoord[2][0], pointsCoord[2][1])))
    lower_left = Feature(geometry=Point((pointsCoord[3][0], pointsCoord[3][1])))
    points_collection = FeatureCollection([upper_left, upper_right, lower_right, lower_left])

    with open(os.path.join(path2saveGeoJSON, f'{name_crop_tif.split(".")[0]}.json'), "w") as fh:
        geojson.dump(points_collection, fh)


def png2Tif(name_crop_tif, input_file_path='', output_file_path='', pointsCoord=0):
    head_tail = os.path.split(input_file_path)
    name_file = head_tail[1]

    dataset = rasterio.open(input_file_path, 'r')
    bands = [1, 2, 3, 4]
    data = dataset.read(bands)

    if pointsCoord:
        bbox = [pointsCoord[3][0], pointsCoord[3][1], pointsCoord[1][0], pointsCoord[1][1]]
    else:
        bbox = [0, 0, 0, 0]

    transform = rasterio.transform.from_bounds(
        *bbox, data.shape[1], data.shape[2])
    crs = {'init': 'epsg:32637'}

    with rasterio.open(os.path.join(output_file_path, f'corrected_{name_crop_tif}'), 'w', driver='GTiff',
                       width=data.shape[1], height=data.shape[2],
                       count=4, dtype=data.dtype, nodata=0,
                       transform=transform, crs=crs) as dst:
        dst.write(data, indexes=bands)

    return 0
