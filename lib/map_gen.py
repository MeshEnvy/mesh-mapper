"""Map generation functions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
from typing import Dict
import rasterio

from lib.log import log_info, log_debug, log_warn


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

