"""DEM data handling and coordinate transformations."""

import math
import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from typing import Tuple

from lib.log import log_info, log_debug, log_error

# Don't import elevation at module level - import it in the function after setting cache



def download_dem_data(bounds: Tuple[float, float, float, float], 
                      output_path: str = "dem_data.tif",
                      resolution: int = 30,
                      base_cache_dir: str = ".cache") -> str:
    """
    Download SRTM DEM data for the specified bounds using elevation library.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        output_path: Path to save DEM data (MUST be absolute path)
        resolution: 30 or 90 meters
        base_cache_dir: Base cache directory (SRTM tiles will be cached in <base_cache_dir>/srtm/)
    
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
    
    # Configure elevation library cache directory to .cache/srtm/
    srtm_cache_dir = os.path.abspath(os.path.join(base_cache_dir, 'srtm'))
    os.makedirs(srtm_cache_dir, exist_ok=True)
    
    # Set elevation cache directory via environment variable BEFORE importing elevation
    # The elevation library checks ELEVATION_CACHE environment variable when it initializes
    # This must be set before any import of elevation or elevation.io modules
    original_cache = os.environ.get('ELEVATION_CACHE')
    os.environ['ELEVATION_CACHE'] = srtm_cache_dir
    log_info(f"SRTM tiles will be cached in: {srtm_cache_dir}")
    log_debug(f"ELEVATION_CACHE environment variable set to: {srtm_cache_dir}")
    
    try:
        # Import elevation AFTER setting environment variable
        # This ensures the cache directory is used during initialization
        import elevation
        
        # Also try to configure via elevation.io if available and patch if needed
        try:
            import elevation.io
            # Try to patch the cache directory function if it exists
            if hasattr(elevation.io, 'get_cache_dir'):
                # Store original function
                original_get_cache = elevation.io.get_cache_dir
                # Create patched version that returns our cache directory
                def patched_get_cache():
                    return srtm_cache_dir
                elevation.io.get_cache_dir = patched_get_cache
                log_debug("Patched elevation.io.get_cache_dir() to use custom cache")
        except (ImportError, AttributeError):
            # elevation.io might not be available or have different structure
            pass
        
        # Use elevation library to download SRTM data
        # output_path MUST be an absolute path
        log_debug(f"Starting elevation.clip() with output path: {output_path}")
        elevation.clip(bounds=bounds, output=output_path, product=product)
        
    finally:
        # Restore original cache setting if it existed
        if original_cache is not None:
            os.environ['ELEVATION_CACHE'] = original_cache
        elif 'ELEVATION_CACHE' in os.environ:
            del os.environ['ELEVATION_CACHE']
    
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


def get_dem_bounds(dem_path: str) -> Tuple[float, float, float, float]:
    """
    Get the geographic bounds of a DEM file.
    
    Returns:
        (min_lon, min_lat, max_lon, max_lat)
    """
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        return (bounds.left, bounds.bottom, bounds.right, bounds.top)


def dem_covers_bounds(dem_path: str, required_bounds: Tuple[float, float, float, float]) -> bool:
    """
    Check if an existing DEM file covers the required bounding box.
    
    Args:
        dem_path: Path to DEM file
        required_bounds: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        True if DEM covers the required bounds, False otherwise
    """
    try:
        dem_bounds = get_dem_bounds(dem_path)
        req_min_lon, req_min_lat, req_max_lon, req_max_lat = required_bounds
        dem_min_lon, dem_min_lat, dem_max_lon, dem_max_lat = dem_bounds
        
        # Check if DEM bounds fully contain required bounds
        covers = (dem_min_lon <= req_min_lon and 
                 dem_min_lat <= req_min_lat and 
                 dem_max_lon >= req_max_lon and 
                 dem_max_lat >= req_max_lat)
        
        log_debug(f"DEM bounds check: DEM={dem_bounds}, Required={required_bounds}, Covers={covers}")
        return covers
    except Exception as e:
        log_debug(f"Error checking DEM bounds: {e}, assuming no coverage")
        return False


def resample_dem(input_path: str, output_path: str, target_resolution_m: int) -> str:
    """
    Resample a DEM file to a different resolution.
    
    Args:
        input_path: Path to source DEM file
        output_path: Path to save resampled DEM (MUST be absolute path)
        target_resolution_m: Target resolution in meters
    
    Returns:
        Path to resampled DEM file
    """
    log_info(f"Resampling DEM from {input_path} to {target_resolution_m}m resolution...")
    log_debug(f"Input: {input_path}, Output: {output_path}, Target resolution: {target_resolution_m}m")
    
    with rasterio.open(input_path) as src:
        bounds = src.bounds
        
        # Calculate target resolution in degrees
        # At the center latitude, 1 degree ≈ 111,320 meters
        center_lat = (bounds.top + bounds.bottom) / 2.0
        # Approximate meters per degree at this latitude
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * abs(math.cos(math.radians(center_lat)))
        
        # Resolution in degrees (use average for square pixels)
        resolution_deg = target_resolution_m / meters_per_deg_lat
        
        # Calculate new dimensions
        width = int((bounds.right - bounds.left) / resolution_deg)
        height = int((bounds.top - bounds.bottom) / resolution_deg)
        
        # Create new transform starting from top-left corner
        new_transform = rasterio.Affine(
            resolution_deg,  # pixel width in degrees
            0.0,              # rotation
            bounds.left,      # top-left x (longitude)
            0.0,              # rotation
            -resolution_deg, # pixel height (negative for north-up)
            bounds.top        # top-left y (latitude)
        )
        
        log_debug(f"Original shape: {src.shape}, New shape: ({height}, {width})")
        log_debug(f"Original transform: {src.transform}")
        log_debug(f"New transform: {new_transform}")
        
        # Create output dataset
        profile = src.profile.copy()
        profile.update({
            'width': width,
            'height': height,
            'transform': new_transform,
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            # Reproject using bilinear resampling
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=dst.crs,
                resampling=Resampling.bilinear,
            )
    
    log_info(f"DEM resampled and saved to: {output_path}")
    log_debug(f"Resampling complete")
    return output_path

