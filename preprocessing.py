import numpy as np

from image_utils import adjust_gamma


def make_4channels_from_tiff(gdal_file):
    final_arr = [[], [], [], []]

    raster_band_r = gdal_file.GetRasterBand(1)
    raster_band_g = gdal_file.GetRasterBand(2)
    raster_band_b = gdal_file.GetRasterBand(3)
    # Nir channel
    raster_band_Nir = gdal_file.GetRasterBand(4)

    band_data_r = raster_band_r.ReadAsArray()
    band_data_g = raster_band_g.ReadAsArray()
    band_data_b = raster_band_b.ReadAsArray()
    band_data_nir = raster_band_Nir.ReadAsArray()

    final_arr[0] = band_data_r
    final_arr[1] = band_data_g
    final_arr[2] = band_data_b
    final_arr[3] = band_data_nir

    final_arr = np.array(final_arr)
    image_res = final_arr.transpose(1, 2, 0)

    image_res = ((image_res - image_res.min()) / (image_res.max() - image_res.min()) * 255.0).astype(np.uint8)
    # rescaled = adjust_gamma(rescaled, gamma=2.0)

    return image_res


def make_png_from_tiff(gdal_file):
    final_arr = [[], [], []]

    raster_band_r = gdal_file.GetRasterBand(1)
    raster_band_g = gdal_file.GetRasterBand(2)
    raster_band_b = gdal_file.GetRasterBand(3)

    # Nir channel
    # raster_band_Nir = gdal_file.GetRasterBand(4)

    band_data_r = raster_band_r.ReadAsArray()
    band_data_g = raster_band_g.ReadAsArray()
    band_data_b = raster_band_b.ReadAsArray()

    final_arr[0] = band_data_r
    final_arr[1] = band_data_g
    final_arr[2] = band_data_b

    final_arr = np.array(final_arr)
    image_res = final_arr.transpose(1, 2, 0)

    rescaled = ((image_res - image_res.min()) / (image_res.max() - image_res.min()) * 255.0).astype(np.uint8)
    rescaled = adjust_gamma(rescaled, gamma=2.0)

    return rescaled
