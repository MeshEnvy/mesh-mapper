#!/usr/bin/env python3
"""
Meshtastic RF Coverage Mapper
Generates terrain-aware RF coverage maps for fixed Meshtastic nodes.

Features:
- Terrain-aware line-of-sight analysis with Fresnel zone clearance
- Configurable LoRa presets (Long-Fast, Medium-Fast, etc.)
- Private map: shows node locations, individual coverage, and overlap zones
- Public map: shows only aggregate coverage area
- Supports 30m and 90m DEM resolution
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import generic_filter
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
import elevation
import folium
from folium import plugins
import argparse
from typing import Dict, List, Tuple
import warnings
import sys
warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

_debug_enabled = False
_silent_enabled = False


def set_logging(debug: bool = False, silent: bool = False):
    """Set global logging flags."""
    global _debug_enabled, _silent_enabled
    _debug_enabled = debug
    _silent_enabled = silent


def log_info(message: str):
    """Log an INFO message."""
    if not _silent_enabled:
        print(f'[INFO] {message}', file=sys.stdout)


def log_debug(message: str):
    """Log a DEBUG message."""
    if not _silent_enabled and _debug_enabled:
        print(f'[DEBUG] {message}', file=sys.stdout)


def log_error(message: str):
    """Log an ERROR message."""
    if not _silent_enabled:
        print(f'[ERROR] {message}', file=sys.stderr)


def log_warn(message: str):
    """Log a WARN message."""
    if not _silent_enabled:
        print(f'[WARN] {message}', file=sys.stdout)


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

LORA_PRESETS = {
    "Long-Fast": {
        "spreading_factor": 11,
        "bandwidth": 250000,  # Hz
        "coding_rate": "4/8",
        "receiver_sensitivity": -134,  # dBm
        "description": "SF11, BW250, CR4/8"
    },
    "Long-Moderate": {
        "spreading_factor": 11,
        "bandwidth": 125000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -137,
        "description": "SF11, BW125, CR4/8"
    },
    "Long-Slow": {
        "spreading_factor": 12,
        "bandwidth": 125000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -140,
        "description": "SF12, BW125, CR4/8"
    },
    "Medium-Fast": {
        "spreading_factor": 10,
        "bandwidth": 250000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -131,
        "description": "SF10, BW250, CR4/8"
    },
    "Medium-Slow": {
        "spreading_factor": 11,
        "bandwidth": 125000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -137,
        "description": "SF11, BW125, CR4/8"
    },
    "Short-Fast": {
        "spreading_factor": 7,
        "bandwidth": 250000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -123,
        "description": "SF7, BW250, CR4/8"
    }
}

DEFAULT_CONFIG = {
    "lora_preset": "Long-Fast",
    "frequency_mhz": 915,
    "tx_power_dbm": 27,  # ~0.5W
    "fade_margin_db": 15,  # Conservative fade margin
    "analysis_radius_miles": 30,
    "dem_resolution_m": 30,  # 30 or 90
    "fresnel_zone_clearance": 0.6,  # 60% of first Fresnel zone
    "earth_radius_km": 6371,
    "antenna_gain_dbi": 2.15,  # Typical omnidirectional antenna
    "num_azimuths": 360,  # Ray-casting resolution (one per degree)
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet * 0.3048


def miles_to_meters(miles: float) -> float:
    """Convert miles to meters."""
    return miles * 1609.344


def meters_to_miles(meters: float) -> float:
    """Convert meters to miles."""
    return meters / 1609.344


def dbm_to_watts(dbm: float) -> float:
    """Convert dBm to watts."""
    return 10 ** ((dbm - 30) / 10)


def watts_to_dbm(watts: float) -> float:
    """Convert watts to dBm."""
    return 10 * np.log10(watts) + 30


def fresnel_radius(distance_m: float, frequency_hz: float, zone: int = 1) -> float:
    """
    Calculate Fresnel zone radius at the midpoint.
    
    Args:
        distance_m: Total distance between transmitter and receiver (meters)
        frequency_hz: Frequency in Hz
        zone: Fresnel zone number (typically 1)
    
    Returns:
        Radius in meters
    """
    wavelength = 3e8 / frequency_hz  # Speed of light / frequency
    d1 = distance_m / 2  # Distance to midpoint
    d2 = distance_m / 2
    
    radius = np.sqrt((zone * wavelength * d1 * d2) / (d1 + d2))
    return radius


def calculate_path_loss_free_space(distance_m: float, frequency_hz: float) -> float:
    """
    Calculate free-space path loss using Friis equation.
    
    Args:
        distance_m: Distance in meters
        frequency_hz: Frequency in Hz
    
    Returns:
        Path loss in dB
    """
    if distance_m <= 0:
        return 0
    
    # Friis free-space path loss formula
    wavelength = 3e8 / frequency_hz
    path_loss = 20 * np.log10(4 * np.pi * distance_m / wavelength)
    
    return path_loss


def calculate_two_ray_path_loss(distance_m: float, frequency_hz: float, 
                                 h_tx: float, h_rx: float = 1.5) -> float:
    """
    Calculate path loss using two-ray ground reflection model.
    More accurate than free-space for ground-based communications.
    
    Args:
        distance_m: Distance in meters
        frequency_hz: Frequency in Hz
        h_tx: Transmitter height above ground (meters)
        h_rx: Receiver height above ground (meters)
    
    Returns:
        Path loss in dB
    """
    if distance_m <= 0:
        return 0
    
    # Critical distance where two-ray model begins to dominate
    critical_distance = (4 * np.pi * h_tx * h_rx) / (3e8 / frequency_hz)
    
    if distance_m < critical_distance:
        # Use free-space model for short distances
        return calculate_path_loss_free_space(distance_m, frequency_hz)
    else:
        # Two-ray model for longer distances
        path_loss = 40 * np.log10(distance_m) - (10 * np.log10(h_tx) + 10 * np.log10(h_rx))
        return path_loss


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points on Earth.
    
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


# ============================================================================
# DEM DATA HANDLING
# ============================================================================

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


def load_dem_data(dem_path: str) -> Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    """
    Load DEM data from file.
    
    Returns:
        (elevation_array, transform, crs)
    """
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


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

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


# ============================================================================
# VIEWSHED AND LINE-OF-SIGHT ANALYSIS
# ============================================================================

def check_line_of_sight(dem: np.ndarray, transform: rasterio.Affine,
                        tx_row: int, tx_col: int, tx_elev: float,
                        rx_row: int, rx_col: int,
                        frequency_hz: float, fresnel_clearance: float = 0.6,
                        earth_radius_km: float = 6371) -> Tuple[bool, float]:
    """
    Check if there is line-of-sight between transmitter and receiver.
    Accounts for terrain, Earth curvature, and Fresnel zone clearance.
    
    Args:
        dem: Digital elevation model array
        transform: Rasterio transform for coordinate conversion
        tx_row, tx_col: Transmitter pixel coordinates
        tx_elev: Transmitter elevation above ground (meters)
        rx_row, rx_col: Receiver pixel coordinates
        frequency_hz: Frequency for Fresnel zone calculation
        fresnel_clearance: Fraction of first Fresnel zone to clear (0.6 = 60%)
        earth_radius_km: Earth radius for curvature correction
    
    Returns:
        (has_los: bool, obstruction_loss_db: float)
    """
    # Get terrain elevations
    tx_terrain = dem[tx_row, tx_col]
    rx_terrain = dem[rx_row, rx_col]
    log_debug(f"Line-of-sight check: tx_terrain={tx_terrain:.1f}m, rx_terrain={rx_terrain:.1f}m")
    
    # Absolute elevations of antennas
    tx_height = tx_terrain + tx_elev
    rx_height = rx_terrain + 1.5  # Assume 1.5m receiver height
    log_debug(f"Antenna heights: tx={tx_height:.1f}m, rx={rx_height:.1f}m")
    
    # Use Bresenham's line algorithm to get all pixels along the path
    path_pixels = bresenham_line(tx_row, tx_col, rx_row, rx_col)
    log_debug(f"Path length: {len(path_pixels)} pixels")
    
    if len(path_pixels) < 2:
        log_debug("Same location, LOS = True")
        return True, 0.0  # Same location
    
    # Calculate distance between endpoints
    tx_lat, tx_lon = pixel_to_latlon(tx_row, tx_col, transform)
    rx_lat, rx_lon = pixel_to_latlon(rx_row, rx_col, transform)
    total_distance = haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
    log_debug(f"Total distance: {total_distance:.1f}m")
    
    if total_distance < 1:  # Less than 1 meter
        log_debug("Distance < 1m, LOS = True")
        return True, 0.0
    
    # Calculate required Fresnel zone clearance
    fresnel_rad = fresnel_radius(total_distance, frequency_hz, zone=1)
    required_clearance = fresnel_clearance * fresnel_rad
    log_debug(f"Fresnel radius: {fresnel_rad:.2f}m, required clearance: {required_clearance:.2f}m")
    
    max_obstruction = 0.0  # Track worst obstruction
    
    # Check each point along the path
    for i, (row, col) in enumerate(path_pixels[1:-1], 1):  # Skip endpoints
        if row < 0 or row >= dem.shape[0] or col < 0 or col >= dem.shape[1]:
            continue
        
        # Distance along path from transmitter
        fraction = i / (len(path_pixels) - 1)
        distance_from_tx = total_distance * fraction
        
        # Expected height of line-of-sight beam (with Earth curvature)
        # Linear interpolation between tx and rx heights
        los_height = tx_height + (rx_height - tx_height) * fraction
        
        # Earth curvature correction (bulge)
        earth_bulge = (distance_from_tx * (total_distance - distance_from_tx)) / (2 * earth_radius_km * 1000)
        los_height -= earth_bulge
        
        # Actual terrain elevation at this point
        terrain_elev = dem[row, col]
        
        # Calculate clearance (can be negative if obstructed)
        clearance = los_height - terrain_elev
        
        # Check if obstruction exceeds Fresnel zone requirement
        obstruction = required_clearance - clearance
        
        if obstruction > max_obstruction:
            max_obstruction = obstruction
            log_debug(f"New max obstruction at pixel ({row},{col}): {max_obstruction:.2f}m")
    
    # Determine if path is viable
    has_los = max_obstruction <= 0
    log_debug(f"LOS result: {has_los}, max_obstruction: {max_obstruction:.2f}m")
    
    # Calculate additional loss due to partial obstruction
    # Using knife-edge diffraction approximation
    obstruction_loss_db = 0.0
    if max_obstruction > 0:
        # Normalized obstruction parameter
        wavelength = wavelength_from_frequency(frequency_hz)
        v = max_obstruction * np.sqrt(2 / (wavelength * (total_distance / 2)))
        log_debug(f"Knife-edge parameter v: {v:.3f}")
        
        if v < -0.8:
            obstruction_loss_db = 0
        else:
            obstruction_loss_db = 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)
        log_debug(f"Obstruction loss: {obstruction_loss_db:.2f} dB")
    
    return has_los, obstruction_loss_db


def wavelength_from_frequency(frequency_hz: float) -> float:
    """Calculate wavelength from frequency."""
    return 3e8 / frequency_hz


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm to get all pixels along a line.
    
    Returns:
        List of (row, col) coordinates
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        points.append((x, y))
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return points


# ============================================================================
# RF COVERAGE CALCULATION
# ============================================================================

def calculate_coverage_map(nodes_df: pd.DataFrame, dem: np.ndarray, 
                          transform: rasterio.Affine, config: Dict) -> Dict:
    """
    Calculate RF coverage for all nodes.
    
    Args:
        nodes_df: DataFrame with columns: node_name, lat, lon, elev
        dem: Digital elevation model array
        transform: Rasterio transform
        config: Configuration dictionary
    
    Returns:
        Dictionary with coverage data for each node
    """
    log_info("="*80)
    log_info("CALCULATING RF COVERAGE")
    log_info("="*80)
    
    # Get LoRa preset parameters
    preset = LORA_PRESETS[config['lora_preset']]
    rx_sensitivity = preset['receiver_sensitivity']
    frequency_hz = config['frequency_mhz'] * 1e6
    log_debug(f"LoRa preset lookup: {config['lora_preset']}")
    log_debug(f"Frequency calculation: {config['frequency_mhz']} MHz = {frequency_hz} Hz")
    
    log_info(f"\nLoRa Preset: {config['lora_preset']}")
    log_info(f"  {preset['description']}")
    log_info(f"  Receiver Sensitivity: {rx_sensitivity} dBm")
    log_info(f"Frequency: {config['frequency_mhz']} MHz")
    log_info(f"TX Power: {config['tx_power_dbm']} dBm")
    log_info(f"Fade Margin: {config['fade_margin_db']} dB")
    log_info(f"Analysis Radius: {config['analysis_radius_miles']} miles")
    
    # Calculate link budget
    # Available signal at receiver = TX power + TX gain + RX gain - Path Loss - Fade Margin
    # For link to close: RX signal must be >= RX sensitivity
    # Therefore: TX power + TX gain + RX gain - Path Loss - Fade Margin >= RX sensitivity
    # Max path loss = TX power + TX gain + RX gain - RX sensitivity - Fade Margin
    
    tx_gain = config['antenna_gain_dbi']
    rx_gain = config['antenna_gain_dbi']  # Assume same antenna on both ends
    log_debug(f"Antenna gains: TX={tx_gain} dBi, RX={rx_gain} dBi")
    
    max_path_loss = (config['tx_power_dbm'] + tx_gain + rx_gain - 
                     rx_sensitivity - config['fade_margin_db'])
    log_debug(f"Link budget calculation: {config['tx_power_dbm']} + {tx_gain} + {rx_gain} - {rx_sensitivity} - {config['fade_margin_db']} = {max_path_loss:.1f} dB")
    
    log_info(f"\nLink Budget:")
    log_info(f"  TX Power: {config['tx_power_dbm']} dBm")
    log_info(f"  TX Gain: {tx_gain} dBi")
    log_info(f"  RX Gain: {rx_gain} dBi")
    log_info(f"  RX Sensitivity: {rx_sensitivity} dBm")
    log_info(f"  Fade Margin: {config['fade_margin_db']} dB")
    log_info(f"  Maximum Path Loss: {max_path_loss:.1f} dB")
    
    # Initialize coverage dictionary
    coverage_data = {}
    log_debug(f"Processing {len(nodes_df)} nodes")
    
    # Process each node
    for idx, node in nodes_df.iterrows():
        log_info(f"\nProcessing node: {node['node_name']}")
        log_info(f"  Location: {node['lat']:.6f}, {node['lon']:.6f}")
        log_info(f"  Elevation: {node['elev']:.1f} ft ({feet_to_meters(node['elev']):.1f} m)")
        log_debug(f"Node index: {idx}")
        
        # Convert node location to pixel coordinates
        tx_row, tx_col = latlon_to_pixel(node['lat'], node['lon'], transform)
        log_debug(f"Node pixel coordinates: ({tx_row}, {tx_col})")
        
        # Check if node is within DEM bounds
        if (tx_row < 0 or tx_row >= dem.shape[0] or 
            tx_col < 0 or tx_col >= dem.shape[1]):
            log_warn(f"Node outside DEM bounds (DEM shape: {dem.shape}), skipping")
            continue
        
        tx_elev_m = feet_to_meters(node['elev'])
        log_debug(f"Transmitter elevation: {tx_elev_m:.2f} m")
        
        # Create coverage mask for this node
        coverage_mask = np.zeros(dem.shape, dtype=bool)
        signal_strength_map = np.full(dem.shape, -200.0, dtype=np.float32)  # Very low default
        log_debug(f"Initialized coverage mask and signal strength map: {dem.shape}")
        
        # Get analysis radius in pixels
        analysis_radius_m = miles_to_meters(config['analysis_radius_miles'])
        log_debug(f"Analysis radius: {config['analysis_radius_miles']} miles = {analysis_radius_m:.1f} m")
        
        # Create bounding box around node
        pixel_size = abs(transform[0])  # Approximate pixel size in degrees
        radius_pixels = int(analysis_radius_m / (pixel_size * 111320))  # Rough conversion
        log_debug(f"Pixel size: {pixel_size:.8f} degrees, radius in pixels: {radius_pixels}")
        
        min_row = max(0, tx_row - radius_pixels)
        max_row = min(dem.shape[0], tx_row + radius_pixels + 1)
        min_col = max(0, tx_col - radius_pixels)
        max_col = min(dem.shape[1], tx_col + radius_pixels + 1)
        log_debug(f"Bounding box: rows [{min_row}, {max_row}), cols [{min_col}, {max_col})")
        
        pixels_to_check = 0
        pixels_with_coverage = 0
        total_pixels = (max_row - min_row) * (max_col - min_col)
        progress_interval = max(1, total_pixels // 20)  # Update every 5%
        log_debug(f"Total pixels to analyze: {total_pixels:,}, progress interval: {progress_interval}")
        
        log_info(f"  Analyzing {total_pixels:,} pixels...")
        
        # Check each pixel in the bounding box
        for rx_row in range(min_row, max_row):
            for rx_col in range(min_col, max_col):
                pixels_to_check += 1
                
                # Show progress
                if pixels_to_check % progress_interval == 0:
                    progress_pct = (pixels_to_check / total_pixels) * 100
                    log_debug(f"Progress: {progress_pct:.0f}% ({pixels_to_check:,}/{total_pixels:,} pixels)")
                
                # Skip if same as transmitter
                if rx_row == tx_row and rx_col == tx_col:
                    coverage_mask[rx_row, rx_col] = True
                    signal_strength_map[rx_row, rx_col] = config['tx_power_dbm']
                    pixels_with_coverage += 1
                    continue
                
                # Calculate distance
                rx_lat, rx_lon = pixel_to_latlon(rx_row, rx_col, transform)
                distance_m = haversine_distance(node['lat'], node['lon'], rx_lat, rx_lon)
                log_debug(f"Pixel ({rx_row},{rx_col}) distance: {distance_m:.1f} m")
                
                # Skip if beyond analysis radius
                if distance_m > analysis_radius_m:
                    log_debug(f"Pixel ({rx_row},{rx_col}) beyond radius, skipping")
                    continue
                
                # Check line of sight
                has_los, obstruction_loss = check_line_of_sight(
                    dem, transform, tx_row, tx_col, tx_elev_m, rx_row, rx_col,
                    frequency_hz, config['fresnel_zone_clearance'], config['earth_radius_km']
                )
                log_debug(f"Pixel ({rx_row},{rx_col}) LOS: {has_los}, obstruction_loss: {obstruction_loss:.2f} dB")
                
                if not has_los:
                    log_debug(f"Pixel ({rx_row},{rx_col}) no LOS, skipping")
                    continue  # No coverage if no line of sight
                
                # Calculate path loss
                path_loss = calculate_two_ray_path_loss(distance_m, frequency_hz, tx_elev_m)
                log_debug(f"Pixel ({rx_row},{rx_col}) path_loss: {path_loss:.2f} dB")
                
                # Add obstruction loss if any
                total_loss = path_loss + obstruction_loss
                log_debug(f"Pixel ({rx_row},{rx_col}) total_loss: {total_loss:.2f} dB")
                
                # Calculate received signal strength
                rx_signal = config['tx_power_dbm'] + tx_gain + rx_gain - total_loss
                log_debug(f"Pixel ({rx_row},{rx_col}) rx_signal: {rx_signal:.2f} dBm, threshold: {rx_sensitivity + config['fade_margin_db']:.2f} dBm")
                
                # Check if signal is above sensitivity threshold (with fade margin)
                if rx_signal >= (rx_sensitivity + config['fade_margin_db']):
                    coverage_mask[rx_row, rx_col] = True
                    signal_strength_map[rx_row, rx_col] = rx_signal
                    pixels_with_coverage += 1
                    log_debug(f"Pixel ({rx_row},{rx_col}) has coverage, signal: {rx_signal:.2f} dBm")
        
        log_debug("Pixel analysis complete")
        coverage_area_km2 = pixels_with_coverage * (pixel_size * 111.32) ** 2
        
        log_info(f"  ✓ Analysis complete!")
        log_info(f"  Pixels checked: {pixels_to_check:,}")
        log_info(f"  Pixels with coverage: {pixels_with_coverage:,}")
        log_info(f"  Coverage area: {coverage_area_km2:.1f} km² ({coverage_area_km2 * 0.386102:.1f} sq mi)")
        log_debug(f"Coverage percentage: {(pixels_with_coverage / pixels_to_check * 100):.2f}%")
        
        coverage_data[node['node_name']] = {
            'lat': node['lat'],
            'lon': node['lon'],
            'elev': node['elev'],
            'row': tx_row,
            'col': tx_col,
            'coverage_mask': coverage_mask,
            'signal_strength': signal_strength_map
        }
        log_debug(f"Coverage data stored for node: {node['node_name']}")
    
    log_info(f"Coverage calculation complete for {len(coverage_data)} nodes")
    return coverage_data


# ============================================================================
# MAP GENERATION
# ============================================================================

def create_overlap_map(coverage_data: Dict) -> np.ndarray:
    """
    Create an overlap count map showing how many nodes cover each pixel.
    
    Returns:
        2D array where each value is the count of overlapping nodes
    """
    if not coverage_data:
        return None
    
    # Get shape from first coverage mask
    first_node = list(coverage_data.values())[0]
    shape = first_node['coverage_mask'].shape
    
    overlap_map = np.zeros(shape, dtype=np.int32)
    
    for node_name, data in coverage_data.items():
        overlap_map += data['coverage_mask'].astype(np.int32)
    
    return overlap_map


def generate_private_map(nodes_df: pd.DataFrame, coverage_data: Dict, 
                        dem: np.ndarray, transform: rasterio.Affine,
                        output_path: str = "coverage_private.png",
                        config: Dict = None):
    """
    Generate private/detailed map showing node locations and coverage overlap.
    
    Different colors for different overlap levels:
    - 1 node: base color
    - 2 nodes: distinct color
    - 3 nodes: another distinct color
    - etc.
    """
    log_info("="*80)
    log_info("GENERATING PRIVATE MAP")
    log_info("="*80)
    
    overlap_map = create_overlap_map(coverage_data)
    log_debug("Created overlap map")
    
    if overlap_map is None:
        log_warn("No coverage data to map")
        return
    
    # Get geographic extent from transform
    height, width = dem.shape
    min_lon, max_lat = transform * (0, 0)
    max_lon, min_lat = transform * (width, height)
    log_debug(f"Map geographic bounds: lon=[{min_lon:.6f}, {max_lon:.6f}], lat=[{min_lat:.6f}, {max_lat:.6f}]")
    
    log_info(f"  Map extent: Lon {min_lon:.4f} to {max_lon:.4f}, Lat {min_lat:.4f} to {max_lat:.4f}")
    
    # Create figure with proper aspect ratio
    log_debug("Creating matplotlib figure (20x16 inches)")
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Show DEM as background (hillshade effect)
    from matplotlib.colors import LightSource
    log_debug("Generating hillshade from DEM")
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=0.05)
    log_debug(f"Hillshade generated: min={np.min(hillshade):.3f}, max={np.max(hillshade):.3f}")
    
    # Display hillshade with proper extent
    ax.imshow(hillshade, cmap='gray', alpha=0.5, 
             extent=[min_lon, max_lon, min_lat, max_lat],
             aspect='auto', origin='upper')
    log_debug("Hillshade displayed")
    
    # Create custom colormap for overlaps
    max_overlap = overlap_map.max()
    log_debug(f"Maximum overlap: {max_overlap} nodes")
    
    colors = ['none', '#4E9B2A', '#FFFF00', '#FF8800', '#FF0000', '#CC0000', '#8800CC']
    n_colors = min(len(colors), max_overlap + 1)
    log_debug(f"Using {n_colors} colors for colormap")
    
    overlap_display = np.ma.masked_where(overlap_map == 0, overlap_map)
    log_debug(f"Coverage pixels: {np.count_nonzero(overlap_display)}")
    
    cmap = mcolors.ListedColormap(colors[:n_colors])
    bounds = list(range(n_colors + 1))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    log_debug("Colormap and normalization created")
    
    # Display coverage with proper extent and higher opacity
    im = ax.imshow(overlap_display, cmap=cmap, norm=norm, alpha=0.7, 
                  interpolation='bilinear',
                  extent=[min_lon, max_lon, min_lat, max_lat],
                  aspect='auto', origin='upper')
    log_debug("Coverage overlay displayed")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(1, n_colors), 
                       label='Number of Overlapping Nodes', 
                       fraction=0.035, pad=0.04, shrink=0.8)
    cbar.set_ticklabels([str(i) for i in range(1, n_colors)])
    cbar.ax.tick_params(labelsize=10)
    log_debug("Colorbar added")
    
    # Plot node locations with geographic coordinates
    log_debug(f"Plotting {len(coverage_data)} node markers")
    for node_name, data in coverage_data.items():
        ax.plot(data['lon'], data['lat'], 'w*', markersize=25, 
               markeredgecolor='black', markeredgewidth=3, zorder=10)
        ax.annotate(node_name, (data['lon'], data['lat']), 
                   xytext=(8, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                            edgecolor='black', alpha=0.95, linewidth=2),
                   fontsize=12, fontweight='bold', zorder=11)
        log_debug(f"Plotted marker for node: {node_name}")
    
    # FORCE axis limits
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    log_debug("Axis limits set")
    
    # Set proper axis labels with degree symbols
    ax.set_xlabel('Longitude (°W)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    
    # Format tick labels to show actual coordinates
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{abs(x):.3f}°W'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°N'))
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=11)
    log_debug("Axis labels and formatting applied")
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='black')
    
    # Calculate and display statistics
    pixel_area_km2 = (abs(transform[0]) * 111.32) ** 2  # Convert degrees to km
    total_coverage_km2 = (overlap_map > 0).sum() * pixel_area_km2
    total_coverage_sqmi = total_coverage_km2 * 0.386102
    log_debug(f"Coverage statistics: {total_coverage_km2:.2f} km², {total_coverage_sqmi:.2f} sq mi")
    
    # Get radius from config if available
    radius_text = ""
    if config and 'analysis_radius_miles' in config:
        radius_text = f" | Analysis Radius: {config['analysis_radius_miles']:.0f} miles"
    
    ax.set_title(f'Meshtastic Coverage Map - Private/Detailed\n'
                f'Node Locations and Overlap Analysis{radius_text}\n'
                f'Total Coverage: {total_coverage_km2:.1f} km² ({total_coverage_sqmi:.1f} sq mi)', 
                fontsize=16, fontweight='bold', pad=20)
    
    log_debug("Saving figure to file")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    log_info(f"Private map saved to: {output_path}")
    log_info(f"  Coverage area: {total_coverage_km2:.1f} km² ({total_coverage_sqmi:.1f} sq mi)")
    log_debug(f"Figure saved successfully, closing")
    plt.close()


def generate_public_map(coverage_data: Dict, dem: np.ndarray, 
                       transform: rasterio.Affine,
                       output_path: str = "coverage_public.png",
                       config: Dict = None):
    """
    Generate public map showing only aggregate coverage area without node locations.
    """
    log_info("="*80)
    log_info("GENERATING PUBLIC MAP")
    log_info("="*80)
    
    overlap_map = create_overlap_map(coverage_data)
    log_debug("Created overlap map for public view")
    
    if overlap_map is None:
        log_warn("No coverage data to map")
        return
    
    # Create binary coverage map (any coverage = True)
    coverage_mask = overlap_map > 0
    log_debug(f"Binary coverage mask created: {coverage_mask.sum()} pixels with coverage")
    
    # Get geographic extent
    height, width = dem.shape
    min_lon, max_lat = transform * (0, 0)
    max_lon, min_lat = transform * (width, height)
    log_debug(f"Map geographic bounds: lon=[{min_lon:.6f}, {max_lon:.6f}], lat=[{min_lat:.6f}, {max_lat:.6f}]")
    
    # Create figure
    log_debug("Creating matplotlib figure (20x16 inches)")
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Show DEM as background
    from matplotlib.colors import LightSource
    log_debug("Generating hillshade from DEM")
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(dem, vert_exag=0.05)
    log_debug("Hillshade generated")
    
    ax.imshow(hillshade, cmap='gray', alpha=0.6,
             extent=[min_lon, max_lon, min_lat, max_lat],
             aspect='auto', origin='upper')
    log_debug("Hillshade displayed")
    
    # Show coverage area in bright RED for visibility
    coverage_display = np.ma.masked_where(~coverage_mask, coverage_mask)
    ax.imshow(coverage_display, cmap='Reds', alpha=0.8, 
             interpolation='bilinear', vmin=0, vmax=1,
             extent=[min_lon, max_lon, min_lat, max_lat],
             aspect='auto', origin='upper')
    log_debug("Coverage overlay displayed (red)")
    
    # FORCE axis limits
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    log_debug("Axis limits set")
    
    # Calculate coverage statistics
    pixel_area_km2 = (abs(transform[0]) * 111.32) ** 2
    coverage_pixels = coverage_mask.sum()
    coverage_area_km2 = coverage_pixels * pixel_area_km2
    coverage_area_sqmi = coverage_area_km2 * 0.386102
    log_debug(f"Coverage statistics: {coverage_area_km2:.2f} km², {coverage_area_sqmi:.2f} sq mi")
    
    # Set proper axis labels
    ax.set_xlabel('Longitude (°W)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=14, fontweight='bold')
    
    # Format tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{abs(x):.3f}°W'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.3f}°N'))
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=11)
    log_debug("Axis labels and formatting applied")
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, color='black')
    
    # Get radius from config if available
    radius_text = ""
    if config and 'analysis_radius_miles' in config:
        radius_text = f" | Analysis Radius: {config['analysis_radius_miles']:.0f} miles"
    
    ax.set_title(f'Meshtastic Coverage Map - Public\n'
                f'Total Coverage Area: {coverage_area_km2:.1f} km² ({coverage_area_sqmi:.1f} sq mi){radius_text}', 
                fontsize=16, fontweight='bold', pad=20)
    
    log_debug("Saving figure to file")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    log_info(f"Public map saved to: {output_path}")
    log_info(f"  Coverage area: {coverage_area_km2:.1f} km² ({coverage_area_sqmi:.1f} sq mi)")
    log_debug("Figure saved successfully, closing")
    plt.close()


def generate_interactive_html_map(nodes_df: pd.DataFrame, coverage_data: Dict,
                                  dem: np.ndarray, transform: rasterio.Affine,
                                  output_path: str = "coverage_interactive.html",
                                  map_type: str = "private"):
    """
    Generate interactive HTML map using Folium.
    
    Args:
        map_type: "private" or "public"
    """
    log_info(f"\nGenerating interactive {map_type} HTML map...")
    log_debug(f"Map type: {map_type}, output path: {output_path}")
    
    if not coverage_data:
        log_warn("No coverage data to map")
        return
    
    # Calculate center point
    lats = [data['lat'] for data in coverage_data.values()]
    lons = [data['lon'] for data in coverage_data.values()]
    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    log_debug(f"Map center: ({center_lat:.6f}, {center_lon:.6f})")
    
    # Create base map
    log_debug("Creating Folium map instance")
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, 
                   tiles='OpenStreetMap')
    
    # Add different tile layers with proper attribution
    log_debug("Adding OpenTopoMap tile layer")
    folium.TileLayer(
        tiles='https://tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='OpenTopoMap'
    ).add_to(m)
    
    if map_type == "private":
        # Add node markers
        log_debug(f"Adding {len(coverage_data)} node markers")
        for node_name, data in coverage_data.items():
            folium.Marker(
                location=[data['lat'], data['lon']],
                popup=f"<b>{node_name}</b><br>Elevation: {data['elev']:.1f} ft",
                tooltip=node_name,
                icon=folium.Icon(color='red', icon='signal', prefix='fa')
            ).add_to(m)
            log_debug(f"Added marker for node: {node_name}")
    
    # Note: Adding raster overlay to Folium is complex and requires converting
    # the coverage raster to GeoJSON or image tiles. For simplicity, we'll add
    # a note indicating that the PNG maps show the detailed coverage.
    
    # Add legend
    log_debug("Adding legend to map")
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><strong>{map_type.title()} Coverage Map</strong></p>
    <p style="font-size:11px">
    View the generated PNG files for detailed<br>
    RF coverage visualization with terrain overlay.
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    log_debug("Adding layer control")
    folium.LayerControl().add_to(m)
    
    # Save map
    log_debug(f"Saving HTML map to: {output_path}")
    m.save(output_path)
    log_info(f"Interactive map saved to: {output_path}")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate RF coverage maps for Meshtastic nodes',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--input', help='CSV file with node data (node_name,lat,lon,elev)', default='nodes.csv')
    parser.add_argument('--config', help='JSON configuration file (optional)')
    parser.add_argument('--lora-preset', choices=list(LORA_PRESETS.keys()),
                       default='Long-Fast', help='LoRa preset to use')
    parser.add_argument('--resolution', type=int, choices=[30, 90], default=30,
                       help='DEM resolution in meters')
    parser.add_argument('--radius', type=float, default=30.0,
                       help='Analysis radius in miles')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--silent', action='store_true',
                       help='Suppress all output')
    
    args = parser.parse_args()
    log_info("//\/\esh⋹nvy")
    
    # Set up logging
    set_logging(debug=args.debug, silent=args.silent)
    log_debug("Debug mode enabled")
    log_debug(f"Command line arguments: {vars(args)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    log_debug(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_debug(f"Output directory created/verified: {output_dir.exists()}")
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    log_debug(f"Initial config loaded: {len(config)} parameters")
    
    if args.config:
        log_info(f"Loading configuration from: {args.config}")
        log_debug(f"Config file path: {Path(args.config).absolute()}")
        try:
            with open(args.config) as f:
                user_config = json.load(f)
                log_debug(f"Config file loaded, keys: {list(user_config.keys())}")
                config.update(user_config)
                log_debug(f"Config merged, final keys: {list(config.keys())}")
        except Exception as e:
            log_error(f"Failed to load config file: {e}")
            raise
    else:
        log_debug("No config file specified, using defaults")
    
    # Override with command-line arguments
    config['lora_preset'] = args.lora_preset
    config['dem_resolution_m'] = args.resolution
    config['analysis_radius_miles'] = args.radius
    log_debug(f"Config overridden with CLI args: preset={args.lora_preset}, resolution={args.resolution}, radius={args.radius}")
    
    log_info("="*80)
    log_info("MESHTASTIC RF COVERAGE MAPPER")
    log_info("="*80)
    log_info(f"\nConfiguration:")
    for key, value in config.items():
        log_info(f"  {key}: {value}")
        log_debug(f"Config item: {key} = {value} (type: {type(value).__name__})")
    
    # Load node data
    log_info(f"\nLoading node data from: {args.input}")
    log_debug(f"CSV file path: {Path(args.input).absolute()}")
    try:
        nodes_df = pd.read_csv(args.input)
        log_debug(f"CSV loaded successfully: {len(nodes_df)} rows, columns: {list(nodes_df.columns)}")
    except Exception as e:
        log_error(f"Failed to load CSV file: {e}")
        raise
    
    log_info(f"Loaded {len(nodes_df)} nodes:")
    for idx, node in nodes_df.iterrows():
        log_info(f"  {node['node_name']}: {node['lat']:.6f}, {node['lon']:.6f}, {node['elev']:.1f} ft")
        log_debug(f"Node {idx}: name={node['node_name']}, lat={node['lat']}, lon={node['lon']}, elev={node['elev']}")
    
    # Calculate bounds for DEM download with generous buffer
    lat_buffer = 0.75  # degrees (~50 miles)
    lon_buffer = 0.75
    log_debug(f"Buffer for DEM bounds: lat={lat_buffer}°, lon={lon_buffer}°")
    
    min_lat = nodes_df['lat'].min() - lat_buffer
    max_lat = nodes_df['lat'].max() + lat_buffer
    min_lon = nodes_df['lon'].min() - lon_buffer
    max_lon = nodes_df['lon'].max() + lon_buffer
    log_debug(f"Node bounds: lat=[{nodes_df['lat'].min():.6f}, {nodes_df['lat'].max():.6f}], "
              f"lon=[{nodes_df['lon'].min():.6f}, {nodes_df['lon'].max():.6f}]")
    
    bounds = (min_lon, min_lat, max_lon, max_lat)
    log_debug(f"DEM bounds with buffer: ({min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f})")
    
    # Download DEM data
    dem_path = output_dir / f"dem_data_{config['dem_resolution_m']}m.tif"
    log_debug(f"DEM path: {dem_path}")
    
    if not dem_path.exists():
        log_debug("DEM file does not exist, downloading")
        # Convert to absolute path for elevation library
        download_dem_data(bounds, str(dem_path.absolute()), config['dem_resolution_m'])
    else:
        log_info(f"\nUsing existing DEM data: {dem_path}")
        log_debug(f"DEM file exists, size: {dem_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load DEM data
    log_info("\nLoading DEM data...")
    dem, transform, crs = load_dem_data(str(dem_path))
    log_info(f"DEM shape: {dem.shape}")
    log_info(f"DEM bounds: {bounds}")
    log_debug(f"DEM loaded: shape={dem.shape}, dtype={dem.dtype}, "
              f"min_elev={np.nanmin(dem):.1f}m, max_elev={np.nanmax(dem):.1f}m")
    
    # Calculate coverage
    log_debug("Starting coverage calculation")
    coverage_data = calculate_coverage_map(nodes_df, dem, transform, config)
    log_debug(f"Coverage calculation complete, {len(coverage_data)} nodes processed")
    
    # Generate maps
    private_map_path = output_dir / "coverage_private.png"
    public_map_path = output_dir / "coverage_public.png"
    private_html_path = output_dir / "coverage_private_interactive.html"
    public_html_path = output_dir / "coverage_public_interactive.html"
    log_debug(f"Output file paths determined: private={private_map_path}, public={public_map_path}")
    
    generate_private_map(nodes_df, coverage_data, dem, transform, str(private_map_path), config)
    generate_public_map(coverage_data, dem, transform, str(public_map_path), config)
    generate_interactive_html_map(nodes_df, coverage_data, dem, transform, 
                                  str(private_html_path), "private")
    generate_interactive_html_map(nodes_df, coverage_data, dem, transform,
                                  str(public_html_path), "public")
    log_debug("All maps generated successfully")
    
    log_info("\n" + "="*80)
    log_info("PROCESSING COMPLETE")
    log_info("="*80)
    log_info(f"\nOutput files:")
    log_info(f"  Private map (PNG): {private_map_path}")
    log_info(f"  Public map (PNG): {public_map_path}")
    log_info(f"  Private map (HTML): {private_html_path}")
    log_info(f"  Public map (HTML): {public_html_path}")
    log_info(f"  DEM data: {dem_path}")
    log_debug("Main function completed successfully")


if __name__ == "__main__":
    main()
