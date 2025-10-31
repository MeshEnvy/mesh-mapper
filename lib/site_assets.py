"""Site asset caching and management."""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

from lib.log import log_debug


def sanitize_node_name(node_name: str) -> str:
    """Sanitize node name for filesystem compatibility."""
    # Replace invalid filesystem characters
    invalid_chars = '<>:"/\\|?*'
    sanitized = node_name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    return sanitized


def build_node_metadata_dict(node_data: pd.Series, config: Dict, dem_path: str) -> Dict:
    """
    Build a unified metadata dictionary from node data, config, and DEM info.
    This is used both for hash computation and metadata.json storage.
    
    Args:
        node_data: Series with node CSV fields
        config: Configuration dictionary
        dem_path: Path to DEM file (required)
    
    Returns:
        Dictionary with node data, config parameters, and DEM hash
    
    Raises:
        ValueError: If dem_path is not provided or file doesn't exist
    """
    if not dem_path:
        raise ValueError("dem_path is required for building node metadata")
    
    dem_path_obj = Path(dem_path)
    if not dem_path_obj.exists():
        raise ValueError(f"DEM file does not exist: {dem_path}")
    
    # Default TX power (fallback if not in config)
    default_tx_power = 27
    
    metadata = {}
    
    # Node CSV fields (all relevant ones - position changes invalidate cache)
    node_fields = ['node_name', 'lat', 'lon', 'elev', 'preset']
    for field in node_fields:
        if field in node_data:
            metadata[field] = float(node_data[field]) if isinstance(node_data[field], (int, float)) else str(node_data[field])
    
    # TX power (if present in CSV or from config)
    if 'tx_power_dbm' in node_data and not pd.isna(node_data.get('tx_power_dbm', pd.NA)):
        metadata['tx_power_dbm'] = float(node_data['tx_power_dbm'])
    else:
        metadata['tx_power_dbm'] = config.get('tx_power_dbm', default_tx_power)
    
    # Relevant config parameters that affect coverage calculation
    config_params = [
        'analysis_radius_miles',
        'fade_margin_db',
        'fresnel_zone_clearance',
        'antenna_gain_dbi',
        'dem_resolution_m',
        'earth_radius_km',
        'num_azimuths',
    ]
    for param in config_params:
        if param in config:
            metadata[param] = config[param]
    
    # DEM file info (MD5 hash of file contents to detect changes)
    # Calculate MD5 hash of DEM file contents
    md5_hash = hashlib.md5()
    with open(dem_path_obj, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5_hash.update(chunk)
    metadata['dem_md5'] = md5_hash.hexdigest()
    # dem_resolution_m already in metadata from config_params above
    
    return metadata


def compute_config_hash(node_data: pd.Series, config: Dict, dem_path: str) -> str:
    """
    Compute stable hash from node data, config, and DEM info.
    
    Args:
        node_data: Series with node CSV fields
        config: Configuration dictionary
        dem_path: Path to DEM file (required)
    
    Returns:
        SHA256 hash as hexadecimal string
    
    Raises:
        ValueError: If dem_path is not provided or file doesn't exist
    """
    # Build unified metadata dict (node + config + DEM hash)
    kv_dict = build_node_metadata_dict(node_data, config, dem_path)
    
    # Serialize with sorted keys for stability
    json_str = json.dumps(kv_dict, sort_keys=True, default=str)
    
    # Compute SHA256 hash
    hash_obj = hashlib.sha256(json_str.encode('utf-8'))
    return hash_obj.hexdigest()


def save_site_assets(node_data: pd.Series, coverage_mask: np.ndarray,
                     signal_strength: np.ndarray, config_hash: str,
                     sites_dir: Path, config: Dict, dem_path: str):
    """
    Save site coverage assets to disk in <sites_dir>/<name>/<config-hash>/ structure.
    
    Args:
        node_data: Series with node CSV fields
        coverage_mask: Boolean array of coverage
        signal_strength: Float array of signal strength
        config_hash: Config hash string
        sites_dir: Path to sites directory
        config: Configuration dictionary
        dem_path: Path to DEM file (required)
    
    Raises:
        ValueError: If dem_path is not provided or file doesn't exist
    """
    node_name = node_data['node_name']
    sanitized_name = sanitize_node_name(node_name)
    # New structure: <sites_dir>/<name>/<config-hash>/
    site_dir = sites_dir / sanitized_name / config_hash
    site_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(str(site_dir / 'coverage_mask.npy'), coverage_mask)
    np.save(str(site_dir / 'signal_strength.npy'), signal_strength)
    
    # Build unified metadata dict (node + config + DEM hash) - same structure used for hash
    # For metadata.json, we only save the node-specific fields, not all config
    full_metadata = build_node_metadata_dict(node_data, config, dem_path)
    # Extract only node fields for human-readable metadata file
    metadata = {
        'node_name': full_metadata['node_name'],
        'lat': full_metadata['lat'],
        'lon': full_metadata['lon'],
        'elev': full_metadata['elev'],
        'preset': full_metadata['preset'],
        'tx_power_dbm': full_metadata.get('tx_power_dbm'),
    }
    
    with open(site_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
    
    # Save config hash
    with open(site_dir / 'config_hash.txt', 'w') as f:
        f.write(config_hash)
    
    log_debug(f"Saved site assets for {node_name} to {site_dir}")


def list_config_variants(site_name: str, sites_dir: Path) -> list:
    """
    List all available config-hash variants for a site.
    
    Args:
        site_name: Name of the site/node
        sites_dir: Path to sites directory
    
    Returns:
        List of config hash strings (empty if none found)
    """
    sanitized_name = sanitize_node_name(site_name)
    site_dir = sites_dir / sanitized_name
    
    if not site_dir.exists():
        return []
    
    # Find all config-hash subdirectories
    variants = []
    for item in site_dir.iterdir():
        if item.is_dir():
            config_hash_path = item / 'config_hash.txt'
            if config_hash_path.exists():
                try:
                    with open(config_hash_path, 'r') as f:
                        hash_str = f.read().strip()
                        variants.append(hash_str)
                except Exception:
                    pass
    
    return variants


def load_site_assets(site_name: str, sites_dir: Path, config_hash: str = None) -> Dict:
    """
    Load cached site assets from disk.
    
    Args:
        site_name: Name of the site/node
        sites_dir: Path to sites directory
        config_hash: Optional config hash to load. If None and multiple variants exist,
                     behavior depends on caller (should be handled by caller)
    
    Returns:
        Dictionary matching structure expected by map generation functions, or None
    """
    sanitized_name = sanitize_node_name(site_name)
    site_base_dir = sites_dir / sanitized_name
    
    if not site_base_dir.exists():
        return None
    
    # If config_hash not provided, try to find single variant
    if config_hash is None:
        variants = list_config_variants(site_name, sites_dir)
        if len(variants) == 0:
            return None
        if len(variants) == 1:
            config_hash = variants[0]
        else:
            # Multiple variants exist, caller should handle selection
            return None
    
    # Load from config-hash subdirectory
    site_dir = site_base_dir / config_hash
    
    # Load metadata
    metadata_path = site_dir / 'metadata.json'
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load numpy arrays
    coverage_mask_path = site_dir / 'coverage_mask.npy'
    signal_strength_path = site_dir / 'signal_strength.npy'
    
    if not coverage_mask_path.exists() or not signal_strength_path.exists():
        return None
    
    coverage_mask = np.load(str(coverage_mask_path))
    signal_strength = np.load(str(signal_strength_path))
    
    # Return in format expected by map generation functions
    return {
        'lat': metadata['lat'],
        'lon': metadata['lon'],
        'elev': metadata['elev'],
        'row': 0,  # Not needed for map generation, will be recalculated if needed
        'col': 0,  # Not needed for map generation, will be recalculated if needed
        'coverage_mask': coverage_mask,
        'signal_strength': signal_strength
    }


def should_recompute_site(node_data: pd.Series, config: Dict, dem_path: str,
                          sites_dir: Path) -> bool:
    """
    Check if site should be recomputed based on cache validity.
    Checks in <sites_dir>/<name>/<config-hash>/ structure.
    
    Args:
        node_data: Series with node CSV fields
        config: Configuration dictionary
        dem_path: Path to DEM file
        sites_dir: Path to sites directory
    
    Returns:
        True if should recompute, False if cache is valid
    """
    node_name = node_data['node_name']
    sanitized_name = sanitize_node_name(node_name)
    
    # Compute current hash
    current_hash = compute_config_hash(node_data, config, dem_path)
    
    # Check in config-hash subdirectory
    site_dir = sites_dir / sanitized_name / current_hash
    
    # If config-hash subdirectory doesn't exist, need to compute
    if not site_dir.exists():
        return True
    
    # Check if required files exist
    required_files = ['metadata.json', 'coverage_mask.npy', 'signal_strength.npy', 'config_hash.txt']
    for filename in required_files:
        if not (site_dir / filename).exists():
            return True
    
    # Verify config hash matches
    config_hash_path = site_dir / 'config_hash.txt'
    try:
        with open(config_hash_path, 'r') as f:
            cached_hash = f.read().strip()
        if cached_hash != current_hash:
            log_debug(f"Config hash mismatch for {node_name}: cached={cached_hash[:8]}..., current={current_hash[:8]}...")
            return True
    except Exception:
        return True
    
    # Cache is valid
    return False

