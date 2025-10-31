"""RF coverage calculation functions."""

import numpy as np
import pandas as pd
import rasterio
from typing import Dict, List, Tuple
from pathlib import Path

from lib.log import log_info, log_debug, log_error, log_warn
from lib.dem_data import latlon_to_pixel, pixel_to_latlon
from lib.site_assets import should_recompute_site, compute_config_hash, save_site_assets


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

LORA_PRESETS = {
    "Long-Fast": {
        "spreading_factor": 11,
        "bandwidth": 250000,  # Hz
        "coding_rate": "4/8",
        "receiver_sensitivity": -134,  # dBm
        "frequency_mhz": 906.875,  # US region, slot 20
        "description": "SF11, BW250, CR4/8"
    },
    "Long-Moderate": {
        "spreading_factor": 11,
        "bandwidth": 125000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -137,
        "frequency_mhz": 907.375,  # US region, slot 21
        "description": "SF11, BW125, CR4/8"
    },
    "Long-Slow": {
        "spreading_factor": 12,
        "bandwidth": 125000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -140,
        "frequency_mhz": 907.875,  # US region, slot 22
        "description": "SF12, BW125, CR4/8"
    },
    "Medium-Fast": {
        "spreading_factor": 10,
        "bandwidth": 250000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -131,
        "frequency_mhz": 913.125,  # US region, slot 45
        "description": "SF10, BW250, CR4/8"
    },
    "Medium-Slow": {
        "spreading_factor": 11,
        "bandwidth": 125000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -137,
        "frequency_mhz": 913.625,  # US region, slot 46
        "description": "SF11, BW125, CR4/8"
    },
    "Short-Fast": {
        "spreading_factor": 7,
        "bandwidth": 250000,
        "coding_rate": "4/8",
        "receiver_sensitivity": -123,
        "frequency_mhz": 918.875,  # US region, slot 68
        "description": "SF7, BW250, CR4/8"
    }
}

DEFAULT_CONFIG = {
    "tx_power_dbm": 27,  # ~0.5W (default, can be overridden per node in CSV)
    "fade_margin_db": 15,  # Conservative fade margin
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
    
    # Absolute elevations of antennas
    tx_height = tx_terrain + tx_elev
    rx_height = rx_terrain + 1.5  # Assume 1.5m receiver height
    
    # Use Bresenham's line algorithm to get all pixels along the path
    path_pixels = bresenham_line(tx_row, tx_col, rx_row, rx_col)
    
    if len(path_pixels) < 2:
        return True, 0.0  # Same location
    
    # Calculate distance between endpoints
    tx_lat, tx_lon = pixel_to_latlon(tx_row, tx_col, transform)
    rx_lat, rx_lon = pixel_to_latlon(rx_row, rx_col, transform)
    total_distance = haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
    
    if total_distance < 1:  # Less than 1 meter
        return True, 0.0
    
    # Calculate required Fresnel zone clearance
    fresnel_rad = fresnel_radius(total_distance, frequency_hz, zone=1)
    required_clearance = fresnel_clearance * fresnel_rad
    
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
    
    # Determine if path is viable
    has_los = max_obstruction <= 0
    
    # Calculate additional loss due to partial obstruction
    # Using knife-edge diffraction approximation
    obstruction_loss_db = 0.0
    if max_obstruction > 0:
        # Normalized obstruction parameter
        wavelength = wavelength_from_frequency(frequency_hz)
        v = max_obstruction * np.sqrt(2 / (wavelength * (total_distance / 2)))
        
        if v < -0.8:
            obstruction_loss_db = 0
        else:
            obstruction_loss_db = 6.9 + 20 * np.log10(np.sqrt((v - 0.1)**2 + 1) + v - 0.1)
    
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
                          transform: rasterio.Affine, config: Dict,
                          sites_dir: Path, dem_path: str) -> Dict:
    """
    Calculate RF coverage for all nodes.
    
    Args:
        nodes_df: DataFrame with columns: node_name, lat, lon, elev, preset, [tx_power_dbm]
        dem: Digital elevation model array
        transform: Rasterio transform
        config: Configuration dictionary
        sites_dir: Path to sites directory for caching
        dem_path: Path to DEM file for cache validation
    
    Returns:
        Dictionary with coverage data for each node
    """
    log_info("="*80)
    log_info("CALCULATING RF COVERAGE")
    log_info("="*80)
    
    log_info(f"Fade Margin: {config['fade_margin_db']} dB")
    log_info(f"Analysis Radius: {config['analysis_radius_miles']} miles")
    
    tx_gain = config['antenna_gain_dbi']
    rx_gain = config['antenna_gain_dbi']  # Assume same antenna on both ends
    log_debug(f"Antenna gains: TX={tx_gain} dBi, RX={rx_gain} dBi")
    
    # Initialize coverage dictionary
    coverage_data = {}
    log_debug(f"Processing {len(nodes_df)} nodes")
    
    # Process each node
    for idx, node in nodes_df.iterrows():
        log_info(f"")
        log_info(f"Processing node: {node['node_name']}")
        log_info(f"  Location: {node['lat']:.6f}, {node['lon']:.6f}")
        log_debug(f"Node index: {idx}")
        
        # Convert node location to pixel coordinates
        tx_row, tx_col = latlon_to_pixel(node['lat'], node['lon'], transform)
        log_debug(f"Node pixel coordinates: ({tx_row}, {tx_col})")
        
        # Check if node is within DEM bounds
        if (tx_row < 0 or tx_row >= dem.shape[0] or 
            tx_col < 0 or tx_col >= dem.shape[1]):
            log_warn(f"Node outside DEM bounds (DEM shape: {dem.shape}), skipping")
            continue
        
        # Get terrain elevation at transmitter location
        tx_terrain_elev_m = dem[tx_row, tx_col]
        tx_elev_m = feet_to_meters(node['elev'])
        total_tx_elev_m = tx_terrain_elev_m + tx_elev_m
        
        log_info(f"  Elevation:")
        log_info(f"    DEM: {tx_terrain_elev_m:.1f} m ({tx_terrain_elev_m * 3.28084:.1f} ft)")
        log_info(f"    TX: {tx_elev_m:.1f} m ({node['elev']:.1f} ft)")
        log_info(f"    Total: {total_tx_elev_m:.1f} m ({total_tx_elev_m * 3.28084:.1f} ft)")
        
        # Check if we should recompute or use cache
        if not should_recompute_site(node, config, str(dem_path), sites_dir):
            log_info(f"  Using cached data for {node['node_name']}")
            # Skip computation - make_map.py will load from cache
            continue
        
        # Get per-node settings from CSV
        preset_name = node['preset']
        if preset_name not in LORA_PRESETS:
            log_error(f"Unknown preset '{preset_name}' for node {node['node_name']}. Available presets: {list(LORA_PRESETS.keys())}")
            continue
        
        preset = LORA_PRESETS[preset_name]
        rx_sensitivity = preset['receiver_sensitivity']
        frequency_mhz = preset['frequency_mhz']
        frequency_hz = frequency_mhz * 1e6
        
        # Get TX power from CSV or use default
        if 'tx_power_dbm' in nodes_df.columns and not pd.isna(node.get('tx_power_dbm', pd.NA)):
            tx_power_dbm = float(node['tx_power_dbm'])
        else:
            tx_power_dbm = config['tx_power_dbm']
        
        log_info(f"  Preset: {preset_name}")
        log_info(f"    {preset['description']}")
        log_info(f"    Frequency: {frequency_mhz} MHz")
        log_info(f"    Receiver Sensitivity: {rx_sensitivity} dBm")
        log_info(f"  TX Power: {tx_power_dbm} dBm")
        
        # Calculate link budget for this node
        max_path_loss = (tx_power_dbm + tx_gain + rx_gain - 
                         rx_sensitivity - config['fade_margin_db'])
        log_debug(f"Link budget calculation: {tx_power_dbm} + {tx_gain} + {rx_gain} - {rx_sensitivity} - {config['fade_margin_db']} = {max_path_loss:.1f} dB")
        
        log_info(f"  Link Budget:")
        log_info(f"    TX Power: {tx_power_dbm} dBm")
        log_info(f"    TX Gain: {tx_gain} dBi")
        log_info(f"    RX Gain: {rx_gain} dBi")
        log_info(f"    RX Sensitivity: {rx_sensitivity} dBm")
        log_info(f"    Fade Margin: {config['fade_margin_db']} dB")
        log_info(f"    Maximum Path Loss: {max_path_loss:.1f} dB")
        
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
                    log_info(f"Progress: {progress_pct:.0f}% ({pixels_to_check:,}/{total_pixels:,} pixels)")
                
                # Skip if same as transmitter
                if rx_row == tx_row and rx_col == tx_col:
                    coverage_mask[rx_row, rx_col] = True
                    signal_strength_map[rx_row, rx_col] = tx_power_dbm
                    pixels_with_coverage += 1
                    continue
                
                # Calculate distance
                rx_lat, rx_lon = pixel_to_latlon(rx_row, rx_col, transform)
                distance_m = haversine_distance(node['lat'], node['lon'], rx_lat, rx_lon)
                
                # Skip if beyond analysis radius
                if distance_m > analysis_radius_m:
                    continue
                
                # Check line of sight
                has_los, obstruction_loss = check_line_of_sight(
                    dem, transform, tx_row, tx_col, tx_elev_m, rx_row, rx_col,
                    frequency_hz, config['fresnel_zone_clearance'], config['earth_radius_km']
                )
                
                if not has_los:
                    continue  # No coverage if no line of sight
                
                # Calculate path loss
                path_loss = calculate_two_ray_path_loss(distance_m, frequency_hz, tx_elev_m)
                
                # Add obstruction loss if any
                total_loss = path_loss + obstruction_loss
                
                # Calculate received signal strength
                rx_signal = tx_power_dbm + tx_gain + rx_gain - total_loss
                
                # Check if signal is above sensitivity threshold (with fade margin)
                if rx_signal >= (rx_sensitivity + config['fade_margin_db']):
                    coverage_mask[rx_row, rx_col] = True
                    signal_strength_map[rx_row, rx_col] = rx_signal
                    pixels_with_coverage += 1
        
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
        
        # Save site assets to cache
        config_hash = compute_config_hash(node, config, str(dem_path))
        save_site_assets(node, coverage_mask, signal_strength_map, config_hash, sites_dir, config, str(dem_path))
        log_info(f"  Saved assets for {node['node_name']} to cache")
    
    log_info(f"Coverage calculation complete for {len(coverage_data)} nodes")
    return coverage_data

