#!/usr/bin/env python3
"""
Meshtastic RF Coverage Mapper CLI
Unified CLI tool for generating terrain-aware RF coverage maps for fixed Meshtastic nodes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import typer
import warnings
warnings.filterwarnings('ignore')

# Import from lib modules
from lib.log import set_logging, log_info, log_debug, log_error, log_warn
from lib.dem_data import download_dem_data, load_dem_data
from lib.coverage import DEFAULT_CONFIG, LORA_PRESETS, calculate_coverage_map
from lib.site_assets import load_site_assets, sanitize_node_name
from lib.map_gen import generate_private_map, generate_public_map

app = typer.Typer()


def get_dem_filename(bounds: tuple) -> str:
    """
    Generate DEM filename from bounds in format: dem_<lat1>_<lng1>_<lat2>_<lng2>.tif
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Filename string
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    # Format: dem_<lat1>_<lng1>_<lat2>_<lng2>.tif
    return f'dem_{min_lat:.2f}_{min_lon:.2f}_{max_lat:.2f}_{max_lon:.2f}.tif'


def find_dem_path(cache_dir: Path, resolution: int, bounds: tuple) -> str:
    """
    Find DEM path for specific bounds in format: dem_<lat1>_<lng1>_<lat2>_<lng2>.tif
    
    Args:
        cache_dir: Path to cache directory
        resolution: Resolution in meters (30 or 90)
        bounds: (min_lon, min_lat, max_lon, max_lat)
    
    Returns:
        Path to DEM file or None if not found
    """
    resolution_dir = cache_dir / str(resolution)
    dem_filename = get_dem_filename(bounds)
    dem_path = resolution_dir / dem_filename
    
    if dem_path.exists():
        log_debug(f'Found DEM for bounds: {dem_path}')
        return str(dem_path)
    
    return None


def prompt_config_selection(site_name: str, variants: list) -> str:
    """
    Prompt user to select a config variant for a site.
    
    Args:
        site_name: Name of the site
        variants: List of config hash strings
    
    Returns:
        Selected config hash string
    """
    print(f"\n{'='*80}")
    print(f'Multiple config variants found for site: {site_name}')
    print(f"{'='*80}")
    print('\nAvailable variants:')
    for idx, variant in enumerate(variants, 1):
        print(f'  {idx}. {variant[:16]}...')
    
    while True:
        try:
            choice = input(f"\nSelect variant (1-{len(variants)}) or 'q' to skip this site: ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice)
            if 1 <= idx <= len(variants):
                return variants[idx - 1]
            else:
                print(f'Invalid choice. Please enter a number between 1 and {len(variants)}.')
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")
        except KeyboardInterrupt:
            print('\nCancelled by user.')
            return None


def load_all_sites(sites_dir: Path) -> dict:
    """
    Load all cached sites from sites directory, handling multiple config variants.
    
    Args:
        sites_dir: Path to sites directory
    
    Returns:
        Dictionary mapping node_name to coverage data
    """
    coverage_data = {}
    
    if not sites_dir.exists():
        log_warn(f'Sites directory does not exist: {sites_dir}')
        return coverage_data
    
    # Find all site directories (now organized as <sites_dir>/<name>/<config-hash>/)
    site_base_dirs = [d for d in sites_dir.iterdir() if d.is_dir()]
    
    if not site_base_dirs:
        log_warn(f'No site directories found in {sites_dir}')
        return coverage_data
    
    log_info(f'Scanning {len(site_base_dirs)} sites from {sites_dir}')
    
    # Group by site name to detect multiple variants
    sites_with_variants = {}
    for site_base_dir in site_base_dirs:
        # Get node name from first metadata.json found in any config-hash subdirectory
        node_name = None
        config_hashes = []
        
        for config_hash_dir in site_base_dir.iterdir():
            if not config_hash_dir.is_dir():
                continue
            
            metadata_path = config_hash_dir / 'metadata.json'
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    if 'node_name' in metadata:
                        node_name = metadata['node_name']
                    # Read config hash
                    config_hash_path = config_hash_dir / 'config_hash.txt'
                    if config_hash_path.exists():
                        with open(config_hash_path, 'r') as f:
                            config_hash = f.read().strip()
                            config_hashes.append(config_hash)
                except Exception:
                    pass
        
        if node_name:
            sites_with_variants[node_name] = {
                'base_dir': site_base_dir,
                'variants': config_hashes
            }
    
    # Load sites, prompting for variant selection if needed
    for node_name, info in sites_with_variants.items():
        variants = info['variants']
        
        if len(variants) == 0:
            log_warn(f'No valid config variants found for {node_name}, skipping')
            continue
        
        config_hash = None
        if len(variants) == 1:
            # Single variant, use it automatically
            config_hash = variants[0]
            log_debug(f'Single variant for {node_name}: {config_hash[:8]}...')
        else:
            # Multiple variants, prompt user
            config_hash = prompt_config_selection(node_name, variants)
            if config_hash is None:
                log_warn(f'Skipping {node_name} (user cancelled or chose to skip)')
                continue
        
        # Load site assets with selected config hash
        try:
            site_data = load_site_assets(node_name, sites_dir, config_hash)
            if site_data is None:
                log_warn(f'Failed to load assets for {node_name}, skipping')
                continue
            
            coverage_data[node_name] = site_data
            log_info(f'Loaded site: {node_name} (variant: {config_hash[:8]}...)')
        except Exception as e:
            log_error(f'Error loading site {node_name}: {e}')
            continue
    
    log_info(f'Successfully loaded {len(coverage_data)} sites')
    return coverage_data


def create_nodes_df_from_sites(coverage_data: dict) -> pd.DataFrame:
    """
    Create a nodes DataFrame from coverage data (for map generation).
    
    Args:
        coverage_data: Dictionary mapping node_name to coverage data
    
    Returns:
        DataFrame with node information
    """
    rows = []
    for node_name, data in coverage_data.items():
        rows.append({
            'node_name': node_name,
            'lat': data['lat'],
            'lon': data['lon'],
            'elev': data['elev'],
        })
    
    return pd.DataFrame(rows)


@app.command()
def analyze(
    input_file: str = typer.Option('nodes.csv', '--input', '-i', help='CSV file with node data (node_name,lat,lon,elev,preset[,tx_power_dbm])'),
    config: str = typer.Option(None, '--config', help='JSON configuration file (optional)'),
    resolution: int = typer.Option(30, '--resolution', help='DEM resolution in meters'),
    radius: float = typer.Option(30.0, '--radius', help='Analysis radius in miles'),
    cache_dir: str = typer.Option('.cache', '--cache-dir', help='Cache directory'),
    min_lat: float = typer.Option(35.0, '--min-lat', help='Minimum latitude for DEM bounding box (default: 35.0 for Nevada)'),
    max_lat: float = typer.Option(42.0, '--max-lat', help='Maximum latitude for DEM bounding box (default: 42.0 for Nevada)'),
    min_lon: float = typer.Option(-120.0, '--min-lon', help='Minimum longitude for DEM bounding box (default: -120.0 for Nevada)'),
    max_lon: float = typer.Option(-114.0, '--max-lon', help='Maximum longitude for DEM bounding box (default: -114.0 for Nevada)'),
    debug: bool = typer.Option(False, '--debug', help='Enable debug output'),
    silent: bool = typer.Option(False, '--silent', help='Suppress all output'),
):
    """Analyze nodes and cache site assets."""
    log_info("//\\/\\esh⋹nvy")
    
    # Set up logging
    set_logging(debug=debug, silent=silent)
    log_debug('Debug mode enabled')
    log_debug(f'Command line arguments: input={input_file}, config={config}, resolution={resolution}, radius={radius}, cache_dir={cache_dir}, bounds=({min_lon}, {min_lat}, {max_lon}, {max_lat}), debug={debug}, silent={silent}')
    
    # Validate resolution
    if resolution not in [30, 90]:
        log_error(f'Resolution must be 30 or 90, got {resolution}')
        raise typer.BadParameter(f'Resolution must be 30 or 90, got {resolution}')
    
    # Create cache directory for DEM files and site assets
    cache_path = Path(cache_dir)
    log_debug(f'Cache directory: {cache_path}')
    cache_path.mkdir(parents=True, exist_ok=True)
    log_debug(f'Cache directory created/verified: {cache_path.exists()}')
    
    # Get resolution from config (will be set below)
    resolution = resolution
    
    # Create resolution-specific directory structure
    resolution_dir = cache_path / str(resolution)
    resolution_dir.mkdir(parents=True, exist_ok=True)
    log_debug(f'Resolution directory: {resolution_dir}')
    
    # Create sites directory for cached site assets (under resolution root)
    sites_dir = resolution_dir / 'sites'
    log_debug(f'Sites directory: {sites_dir}')
    sites_dir.mkdir(parents=True, exist_ok=True)
    log_debug(f'Sites directory created/verified: {sites_dir.exists()}')
    
    # Load configuration
    config_dict = DEFAULT_CONFIG.copy()
    log_debug(f'Initial config loaded: {len(config_dict)} parameters')
    
    if config:
        log_info(f'Loading configuration from: {config}')
        log_debug(f'Config file path: {Path(config).absolute()}')
        try:
            with open(config) as f:
                user_config = json.load(f)
                log_debug(f'Config file loaded, keys: {list(user_config.keys())}')
                config_dict.update(user_config)
                log_debug(f'Config merged, final keys: {list(config_dict.keys())}')
        except Exception as e:
            log_error(f'Failed to load config file: {e}')
            raise
    
    # Override with command-line arguments
    config_dict['dem_resolution_m'] = resolution
    config_dict['analysis_radius_miles'] = radius
    log_debug(f'Config overridden with CLI args: resolution={resolution}, radius={radius}')
    
    log_info("="*80)
    log_info('MESHTASTIC RF COVERAGE MAPPER')
    log_info("="*80)
    log_info(f'\nConfiguration:')
    for key, value in config_dict.items():
        log_info(f'  {key}: {value}')
        log_debug(f'Config item: {key} = {value} (type: {type(value).__name__})')
    
    # Load node data
    log_info(f'\nLoading node data from: {input_file}')
    log_debug(f'CSV file path: {Path(input_file).absolute()}')
    try:
        nodes_df = pd.read_csv(input_file)
        log_debug(f'CSV loaded successfully: {len(nodes_df)} rows, columns: {list(nodes_df.columns)}')
    except Exception as e:
        log_error(f'Failed to load CSV file: {e}')
        raise
    
    # Validate required columns
    required_columns = ['node_name', 'lat', 'lon', 'elev', 'preset']
    missing_columns = [col for col in required_columns if col not in nodes_df.columns]
    if missing_columns:
        log_error(f'CSV file is missing required columns: {missing_columns}')
        log_error(f'Found columns: {list(nodes_df.columns)}')
        log_error(f'Required columns: {required_columns}')
        raise ValueError(f'Missing required columns: {missing_columns}')
    
    # Validate preset values
    invalid_presets = nodes_df[~nodes_df['preset'].isin(LORA_PRESETS.keys())]['preset'].unique()
    if len(invalid_presets) > 0:
        log_error(f'Invalid preset names found in CSV: {list(invalid_presets)}')
        log_error(f'Valid presets: {list(LORA_PRESETS.keys())}')
        raise ValueError(f'Invalid preset names: {list(invalid_presets)}')
    
    log_info(f'Loaded {len(nodes_df)} nodes:')
    for idx, node in nodes_df.iterrows():
        preset_info = f", preset: {node['preset']}"
        tx_power_info = ""
        if 'tx_power_dbm' in nodes_df.columns and not pd.isna(node.get('tx_power_dbm')):
            tx_power_info = f", tx_power: {node['tx_power_dbm']} dBm"
        log_info(f"  {node['node_name']}: {node['lat']:.6f}, {node['lon']:.6f}, {node['elev']:.1f} ft{preset_info}{tx_power_info}")
        log_debug(f"Node {idx}: name={node['node_name']}, lat={node['lat']}, lon={node['lon']}, elev={node['elev']}, preset={node['preset']}")
    
    # Use bounding box from CLI arguments (defaults to Nevada state bounds)
    # Validate bounds
    if min_lat >= max_lat:
        log_error(f'min_lat ({min_lat}) must be less than max_lat ({max_lat})')
        raise typer.BadParameter(f'min_lat must be less than max_lat')
    if min_lon >= max_lon:
        log_error(f'min_lon ({min_lon}) must be less than max_lon ({max_lon})')
        raise typer.BadParameter(f'min_lon must be less than max_lon')
    
    bounds = (min_lon, min_lat, max_lon, max_lat)
    log_info(f'Using DEM bounding box: ({min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f})')
    log_debug(f'DEM bounds: min_lon={bounds[0]:.6f}, min_lat={bounds[1]:.6f}, max_lon={bounds[2]:.6f}, max_lat={bounds[3]:.6f}')
    
    # Download DEM data with filename: dem_<lat1>_<lng1>_<lat2>_<lng2>.tif
    dem_filename = get_dem_filename(bounds)
    dem_path = resolution_dir / dem_filename
    log_debug(f'DEM path: {dem_path}')
    
    if not dem_path.exists():
        log_info(f'\nDEM file not found, downloading for bounds: ({bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f})')
        log_debug('DEM file does not exist, downloading')
        # Convert to absolute path for elevation library
        download_dem_data(bounds, str(dem_path.absolute()), resolution)
    else:
        log_info(f'\nUsing existing DEM data: {dem_path}')
        log_debug(f'DEM file exists, size: {dem_path.stat().st_size / 1024 / 1024:.2f} MB')
    
    # Load DEM data
    log_info('\nLoading DEM data...')
    dem, transform, crs = load_dem_data(str(dem_path))
    log_info(f'DEM shape: {dem.shape}')
    log_info(f'DEM bounds: {bounds}')
    log_debug(f'DEM loaded: shape={dem.shape}, dtype={dem.dtype}, '
              f'min_elev={np.nanmin(dem):.1f}m, max_elev={np.nanmax(dem):.1f}m')
    
    # Calculate coverage and cache site assets
    log_debug('Starting coverage calculation')
    coverage_data = calculate_coverage_map(nodes_df, dem, transform, config_dict, sites_dir, str(dem_path))
    log_debug(f'Coverage calculation complete, {len(coverage_data)} nodes processed')
    
    log_info("\n" + "="*80)
    log_info('PROCESSING COMPLETE')
    log_info("="*80)
    log_info(f'\nSite assets cached to: {sites_dir}')
    log_info(f'  Processed {len(coverage_data)} nodes')
    log_info(f'  DEM data: {dem_path}')
    log_info(f'\nTo generate maps, run:')
    log_info(f'  python mapper.py build --resolution {resolution}')
    log_debug('Analyze command completed successfully')


@app.command()
def build(
    resolution: int = typer.Option(..., '--resolution', help='DEM resolution in meters (30 or 90)'),
    maps_dir: str = typer.Option('maps', '--maps-dir', help='Output directory for maps'),
    cache_dir: str = typer.Option('.cache', '--cache-dir', help='Cache directory'),
    min_lat: float = typer.Option(35.0, '--min-lat', help='Minimum latitude for DEM bounding box (default: 35.0 for Nevada)'),
    max_lat: float = typer.Option(42.0, '--max-lat', help='Maximum latitude for DEM bounding box (default: 42.0 for Nevada)'),
    min_lon: float = typer.Option(-120.0, '--min-lon', help='Minimum longitude for DEM bounding box (default: -120.0 for Nevada)'),
    max_lon: float = typer.Option(-114.0, '--max-lon', help='Maximum longitude for DEM bounding box (default: -114.0 for Nevada)'),
    debug: bool = typer.Option(False, '--debug', help='Enable debug output'),
    silent: bool = typer.Option(False, '--silent', help='Suppress all output'),
):
    """Generate maps from cached site assets."""
    # Set up logging
    set_logging(debug=debug, silent=silent)
    log_debug('Debug mode enabled')
    log_debug(f'Command line arguments: resolution={resolution}, maps_dir={maps_dir}, cache_dir={cache_dir}, debug={debug}, silent={silent}')
    
    # Validate resolution
    if resolution not in [30, 90]:
        log_error(f'Resolution must be 30 or 90, got {resolution}')
        raise typer.BadParameter(f'Resolution must be 30 or 90, got {resolution}')
    
    # Validate bounds
    if min_lat >= max_lat:
        log_error(f'min_lat ({min_lat}) must be less than max_lat ({max_lat})')
        raise typer.BadParameter(f'min_lat must be less than max_lat')
    if min_lon >= max_lon:
        log_error(f'min_lon ({min_lon}) must be less than max_lon ({max_lon})')
        raise typer.BadParameter(f'min_lon must be less than max_lon')
    
    bounds = (min_lon, min_lat, max_lon, max_lat)
    
    # Convert to Path objects
    cache_path = Path(cache_dir)
    resolution_dir = cache_path / str(resolution)
    sites_dir = resolution_dir / 'sites'  # <cache-dir>/<resolution>/sites/
    maps_path = Path(maps_dir)
    
    log_info("="*80)
    log_info('MAP GENERATION FROM CACHED SITES')
    log_info("="*80)
    log_info(f'Resolution: {resolution}m')
    log_info(f'Cache directory: {cache_path}')
    log_info(f'DEM bounding box: ({min_lon:.6f}, {min_lat:.6f}, {max_lon:.6f}, {max_lat:.6f})')
    
    # Find DEM path from cache directory
    log_info('\nLocating DEM file...')
    dem_path_str = find_dem_path(cache_path, resolution, bounds)
    if not dem_path_str:
        log_error(f'Could not find DEM file for bounds in {resolution_dir}')
        log_error('Run mapper.py analyze first to download DEM data')
        return
    
    # Load all sites
    log_info(f'\nLoading sites from: {sites_dir}')
    coverage_data = load_all_sites(sites_dir)
    
    if not coverage_data:
        log_error('No sites loaded. Cannot generate maps.')
        return
    
    log_info(f'Using DEM: {dem_path_str}')
    
    # Load DEM data
    log_info('\nLoading DEM data...')
    try:
        dem, transform, crs = load_dem_data(dem_path_str)
        log_info(f'DEM shape: {dem.shape}')
        log_debug(f'DEM loaded: shape={dem.shape}, dtype={dem.dtype}')
    except Exception as e:
        log_error(f'Failed to load DEM: {e}')
        return
    
    # Create nodes DataFrame for map generation
    nodes_df = create_nodes_df_from_sites(coverage_data)
    
    # Create output directories
    private_dir = maps_path / 'private'
    public_dir = maps_path / 'public'
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    
    private_map_path = private_dir / 'coverage_private.png'
    public_map_path = public_dir / 'coverage_public.png'
    
    log_info(f'\nGenerating maps to {maps_path}...')
    
    # Generate private map
    log_info(f'\nGenerating private map: {private_map_path}')
    try:
        generate_private_map(nodes_df, coverage_data, dem, transform,
                            str(private_map_path), config=None)
    except Exception as e:
        log_error(f'Failed to generate private map: {e}')
        return
    
    # Generate public map
    log_info(f'\nGenerating public map: {public_map_path}')
    try:
        generate_public_map(coverage_data, dem, transform,
                          str(public_map_path), config=None)
    except Exception as e:
        log_error(f'Failed to generate public map: {e}')
        return
    
    log_info("\n" + "="*80)
    log_info('MAP GENERATION COMPLETE')
    log_info("="*80)
    log_info(f'\nOutput files:')
    log_info(f'  Private map: {private_map_path}')
    log_info(f'  Public map: {public_map_path}')
    log_debug('Build command completed successfully')


def main():
    app()


if __name__ == "__main__":
    app()
