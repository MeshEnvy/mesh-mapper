# Meshtastic RF Coverage Mapper

Terrain-aware RF coverage analysis tool for fixed Meshtastic node networks.

## Features

- **Terrain-aware line-of-sight analysis** with Fresnel zone clearance
- **Automatic DEM data download** from SRTM (30m or 90m resolution)
- **Configurable LoRa presets** (Long-Fast, Medium-Fast, Short-Fast, etc.)
- **Two map types**:
  - **Private/Detailed**: Shows node locations, individual coverage, and overlap zones with distinct colors
  - **Public/Basic**: Shows aggregate coverage area only, without node locations
- **Multiple output formats**: Static PNG images and interactive HTML maps
- **Link budget calculations** based on TX power, frequency, and LoRa parameters

## Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install numpy pandas matplotlib scipy rasterio elevation folium --break-system-packages
```

Or create a virtual environment (recommended)

### Additional System Dependencies

The `elevation` library requires `gdal`. On Ubuntu/Debian:

```bash
sudo apt-get install gdal-bin python3-gdal
```

On macOS:

```bash
brew install gdal
```

(May want to get away from this, it's a mess)

## Usage

### Basic Usage

1. Create a CSV file with your node data (see format below)
2. Run the script:

```bash
python meshtastic_coverage_mapper.py nodes.csv
```

This will:

- Download DEM data for the region
- Calculate RF coverage for each node
- Generate private and public maps (PNG and HTML)

### Command-Line Options

```bash
python meshtastic_coverage_mapper.py nodes.csv [options]

Options:
  --resolution {30,90}  DEM resolution in meters (default: 30)
  --radius FLOAT        Analysis radius in miles (default: 30.0)
  --output-dir PATH     Output directory (default: output)
  --config FILE         JSON configuration file for advanced settings
```

### Examples

**Use 90m resolution:**

```bash
python meshtastic_coverage_mapper.py nodes.csv --resolution 90
```

**Analyze 50-mile radius around each node:**

```bash
python meshtastic_coverage_mapper.py nodes.csv --radius 50
```

**Use custom configuration file:**

```bash
python meshtastic_coverage_mapper.py nodes.csv --config my_config.json
```

## Input File Format

### Node Data CSV

Create a CSV file with the following columns:

```csv
node_name,lat,lon,elev,preset
Reno_Peak,39.5296,-119.8138,8200,Long-Fast
Virginia_City,39.3097,-119.6505,6500,Long-Fast
Mount_Rose,39.3436,-119.9122,10776,Medium-Fast
```

Required columns:

- **node_name**: Unique identifier for the node (used in map labels)
- **lat**: Latitude in decimal degrees
- **lon**: Longitude in decimal degrees
- **elev**: Antenna elevation in **feet** (height above sea level, not antenna height above ground)
- **preset**: LoRa preset name (must match one of the predefined presets)

Optional columns:

- **tx_power_dbm**: Transmit power in dBm (defaults to value from config file, typically 27 dBm)

**Note:** Elevation is in feet, not meters. The script converts internally.

**Note:** Each node can use a different preset, allowing you to model heterogeneous networks with nodes using different frequency/modulation settings.

### Configuration File (Optional)

For advanced customization, create a JSON configuration file:

```json
{
  "tx_power_dbm": 27,
  "fade_margin_db": 15,
  "analysis_radius_miles": 30,
  "dem_resolution_m": 30,
  "fresnel_zone_clearance": 0.6,
  "antenna_gain_dbi": 2.15
}
```

**Note:** `lora_preset` and `frequency_mhz` are now specified per-node in the CSV file. The frequency for each preset is automatically determined based on Meshtastic's preset-to-frequency mapping.

## LoRa Presets

The script includes the following Meshtastic LoRa presets with their default frequencies (US region):

| Preset        | Frequency (MHz) | SF  | BW (kHz) | RX Sensitivity | Typical Range |
| ------------- | --------------: | --- | -------: | -------------: | ------------: |
| Long-Fast     |         906.875 | 11  |      250 |       -134 dBm |      15-30 mi |
| Long-Moderate |         907.375 | 11  |      125 |       -137 dBm |      20-35 mi |
| Long-Slow     |         907.875 | 12  |      125 |       -140 dBm |      25-40 mi |
| Medium-Fast   |         913.125 | 10  |      250 |       -131 dBm |      10-20 mi |
| Medium-Slow   |         913.625 | 11  |      125 |       -137 dBm |      15-25 mi |
| Short-Fast    |         918.875 | 7   |      250 |       -123 dBm |       5-10 mi |

**Note:** Actual range depends heavily on terrain, antenna height, and local RF environment.

**Note:** Each preset maps to a specific frequency slot in Meshtastic. The frequencies shown are for the US region and are automatically used when you specify the preset name in the CSV file. For custom presets or other regions, you can add them to the `LORA_PRESETS` dictionary in the code.

## Output Files

The script generates the following files in the output directory:

1. **coverage_private.png**: High-resolution private map showing:

   - Node locations with labels
   - Individual node coverage areas
   - Overlap zones with distinct colors (1 node = green, 2 nodes = yellow, 3 nodes = orange, etc.)
   - Terrain hillshade background

2. **coverage_public.png**: High-resolution public map showing:
   - Aggregate coverage area (single color)
   - No node locations
   - Terrain hillshade background
   - Total coverage area statistics
     **_ JUNK _**
3. **coverage_private_interactive.html**: Interactive Leaflet map with:

   - Node markers with pop-ups
   - Multiple base layer options (OpenStreetMap, Terrain, Topo)
   - Zoom and pan capabilities

4. **coverage_public_interactive.html**: Interactive map without node markers
   **_ /JUNK _**
5. **dem_data_30m.tif** (or **dem_data_90m.tif**): Downloaded terrain elevation data (cached for reuse)

## Technical Details

### RF Propagation Model

The script uses a comprehensive RF propagation model:

1. **Link Budget Calculation**:

   - Accounts for TX power, antenna gains, path loss, and fade margin
   - Compares received signal strength to LoRa preset sensitivity

2. **Path Loss Model**:

   - **Two-ray ground reflection model** for realistic ground-based propagation
   - Falls back to free-space model for short distances
   - More accurate than simple Friis equation

3. **Line-of-Sight Analysis**:

   - Ray-casting algorithm tests terrain obstruction along path
   - **Fresnel zone clearance**: Requires 60% of first Fresnel zone to be clear
   - **Earth curvature correction**: Accounts for radio horizon
   - **Knife-edge diffraction**: Calculates additional loss from partial obstructions

4. **Terrain Integration**:
   - Uses SRTM digital elevation model data
   - Bresenham's line algorithm for efficient path sampling
   - Interpolates LOS beam height considering Earth curvature

### Coverage Calculation Process

For each node:

1. Create bounding box around node (analysis radius)
2. For each pixel in the box:
   - Check if within analysis radius (Haversine distance)
   - Test line-of-sight with Fresnel zone clearance
   - Calculate path loss (two-ray model)
   - Calculate received signal strength
   - Mark as covered if signal > sensitivity + fade margin

### Overlap Detection

The private map uses a multi-level overlap visualization:

- Counts how many nodes can reach each location
- Assigns distinct colors based on overlap count
- Green = single node, Yellow = 2 nodes, Orange = 3 nodes, etc.

## Configuration Parameters

### Key Parameters Explained

- **tx_power_dbm** (27): Transmit power in dBm (~0.5W = 27 dBm)
  (may be too generous for most of these nodes)
- **fade_margin_db** (15): Additional margin for reliability (accounts for multipath fading, weather, etc.)
- **fresnel_zone_clearance** (0.6): Fraction of first Fresnel zone required to be clear (0.6 = 60%)
- **antenna_gain_dbi** (2.15): Omnidirectional antenna gain in dBi
- **dem_resolution_m** (30 or 90): Terrain data resolution in meters
  - 30m: More accurate but larger data download and slower processing
  - 90m: Faster processing, adequate for most analyses

### Adjusting for Different Scenarios

**For more conservative (reliable) coverage estimates:**

- Increase `fade_margin_db` to 20 or 25
- Increase `fresnel_zone_clearance` to 0.8

**For more optimistic coverage estimates:**

- Decrease `fade_margin_db` to 10
- Decrease `fresnel_zone_clearance` to 0.4

**For different antenna types:**

- Directional antennas: Increase `antenna_gain_dbi` (e.g., 10-15 dBi for Yagi)
- Low-gain antennas: Decrease `antenna_gain_dbi` (e.g., 0 dBi for rubber duck)

## Limitations and Considerations

1. **Compute Time**:

   - 30m resolution over large areas can take 10-30 minutes per node
   - Use 90m resolution for faster initial testing

2. **DEM Data Download**:

   - First run downloads terrain data (For Nevada, I grab Tonopah and north, then E-W from CA border to Utah border)
   - Data is cached locally after; it's static

3. **Model Accuracy**:

   - Does not account for vegetation, buildings, or RFI
   - Assumes clear weather conditions - not sure how to account for ducting, heavy snow, etc.
   - Real-world performance may vary ±20-30% from predictions (at least)

4. **Memory Usage**:

   - Large coverage areas with 30m resolution require significant RAM
   - Consider breaking very large regions into smaller sections

5. **Antenna Height**:
   - Script assumes fixed antenna height above ground (configured in `tx_elev_m`)
   - Actual antenna installation height above terrain is critical for performance
   - The elevation in the CSV should be the antenna's elevation above sea level

## Troubleshooting

### Script runs slowly

- Use 90m resolution instead of 30m
- Reduce analysis radius
- Reduce number of nodes being analyzed

### Coverage seems too optimistic/pessimistic

- Adjust `fade_margin_db` in configuration
- Verify antenna elevations are correct (in feet, not meters)
- Check that node coordinates are accurate

## References

- [Meshtastic Documentation](https://meshtastic.org/)
- [LoRa Range Calculator](https://www.rfwireless-world.com/calculators/LoRa-Range-Calculator.html)
- [Fresnel Zone Clearance](https://en.wikipedia.org/wiki/Fresnel_zone)
- [SRTM Data](https://www2.jpl.nasa.gov/srtm/)

## License

This script is provided as-is, do what you want with it.

## Support

For issues or questions, please review the troubleshooting section above,
or contact edwardmitchell73@gmail.com. I'll eventually check email.
