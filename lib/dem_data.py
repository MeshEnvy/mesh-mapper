"""DEM data handling and coordinate transformations."""

import rasterio
import elevation
from typing import Tuple

from lib.log import log_info, log_debug, log_error



def download_dem_data(bounds: Tuple[float, float, float, float], 
                      output_path: str = "dem_data.tif",
                      resolution: int = 30) -> str:
    """
    Download SRTM DEM data for the specified bounds using elevation library.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        output_path: Path to save DEM data (MUST be absolute path)
        resolution: 30 or 90 meters
    
    Returns:
        Path to downloaded DEM file
    """
    log_info(f"Downloading {resolution}m DEM data for bounds: {bounds}")
    log_debug(f"DEM download bounds: min_lon={bounds[0]:.6f}, min_lat={bounds[1]:.6f}, max_lon={bounds[2]:.6f}, max_lat={bounds[3]:.6f}")
    log_info("This may take several minutes depending on area size...")
    
    # Map resolution to SRTM product names
    # SRTM1 = 1 arc-second (~30m)
    # SRTM3 = 3 arc-second (~90m)
    product_map = {
        30: 'SRTM1',
        90: 'SRTM3'
    }
    
    if resolution not in product_map:
        log_error(f"Resolution must be 30 or 90, got {resolution}")
        raise ValueError(f"Resolution must be 30 or 90, got {resolution}")
    
    product = product_map[resolution]
    log_debug(f"Using SRTM product: {product}")
    
    # Use elevation library to download SRTM data
    # output_path MUST be an absolute path
    log_debug(f"Starting elevation.clip() with output path: {output_path}")
    elevation.clip(bounds=bounds, output=output_path, product=product)
    
    log_info(f"DEM data downloaded to: {output_path}")
    log_debug(f"DEM download complete, file size check pending")
    return output_path


def load_dem_data(dem_path: str):
    """
    Load DEM data from file.
    
    Returns:
        (elevation_array, transform, crs)
    """
    import numpy as np
    
    log_debug(f"Opening DEM file: {dem_path}")
    with rasterio.open(dem_path) as src:
        elevation_data = src.read(1)
        transform = src.transform
        crs = src.crs
        log_debug(f"DEM loaded - shape: {elevation_data.shape}, dtype: {elevation_data.dtype}, "
                  f"min: {np.nanmin(elevation_data):.1f}, max: {np.nanmax(elevation_data):.1f}")
        log_debug(f"DEM transform: {transform}")
        log_debug(f"DEM CRS: {crs}")
    
    return elevation_data, transform, crs


def latlon_to_pixel(lat: float, lon: float, transform: rasterio.Affine) -> Tuple[int, int]:
    """
    Convert lat/lon to pixel coordinates in DEM raster.
    
    Returns:
        (row, col) pixel coordinates
    """
    col, row = ~transform * (lon, lat)
    return int(row), int(col)


def pixel_to_latlon(row: int, col: int, transform: rasterio.Affine) -> Tuple[float, float]:
    """
    Convert pixel coordinates to lat/lon.
    
    Returns:
        (lat, lon)
    """
    lon, lat = transform * (col, row)
    return lat, lon

