# Quick Start Guide

## 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
which python
which pip
pip install -r requirements.txt --break-system-packages
```

Or if you prefer a virtual environment, use conda or venv (recommend one of these)

## 2. Prepare Your Node Data

Create a CSV file named `my_nodes.csv` with your actual nodes:

```csv
node_name,lat,lon,elev
Node1,39.5296,-119.8138,8200
Node2,39.3097,-119.6505,6500
Node3,39.3436,-119.9122,10776
```

**Important**:

- Elevation must be in **feet** (not meters)
- Elevation is the antenna height above sea level
- Use decimal degrees for lat/lon

## 3. Run the Script

Basic usage (uses Long-Fast preset, 30m resolution, 30-mile radius):

```bash
python meshtastic_coverage_mapper.py my_nodes.csv
```

## 4. View Results

Check the `output/` directory for:

- `coverage_private.png` - Detailed map with node locations
- `coverage_public.png` - Public map without node locations
- `coverage_private_interactive.html` - Interactive private map \*\*\* JUNK
- `coverage_public_interactive.html` - Interactive public map \*\*\* JUNK

## Example Command-Line Options

**Use different LoRa preset:**

```bash
python meshtastic_coverage_mapper.py my_nodes.csv --lora-preset Long-Slow
```

**Use 90m resolution (faster processing):**

```bash
python meshtastic_coverage_mapper.py my_nodes.csv --resolution 90
```

**Analyze 50-mile radius:**

```bash
python meshtastic_coverage_mapper.py my_nodes.csv --radius 50
```

**Combine multiple options:**

```bash
python meshtastic_coverage_mapper.py my_nodes.csv \
  --lora-preset Long-Slow \
  --resolution 90 \
  --radius 40 \
  --output-dir my_analysis
```

## Testing with Sample Data

Test the script with the included sample data:

```bash
python meshtastic_coverage_mapper.py sample_nodes.csv --resolution 90
```

This uses faster 90m resolution for a quick test run.

## What to Expect

### First Run

- Downloads DEM terrain data (may take 2-10 minutes depending on area size)
- Calculates coverage (5-30 minutes depending on resolution and number of nodes)
- Generates all map outputs

### Subsequent Runs

- Uses cached DEM data (much faster)
- Only recalculates coverage if you change nodes or settings

### Private Map (coverage_private.png)

- Shows exact node locations with names
- Different colors indicate overlap:
  - Green: Single node coverage
  - Yellow: 2 nodes overlap
  - Orange: 3 nodes overlap
  - Red: 4+ nodes overlap
- Use this for planning and network optimization

### Public Map (coverage_public.png)

- Shows only the coverage area
- No node locations revealed
- Single color for all coverage
- Use this for sharing with the public

## Common Issues

**Script is slow:**

- Use `--resolution 90` for faster processing
- Reduce `--radius` value
- Test with fewer nodes first

**No coverage shown:**

- Verify node coordinates are correct
- Check elevation values (should be in feet)
- Try sample data first to verify setup

## Next Steps

1. Review the detailed README.md for all options
2. Adjust configuration in config_example.json for custom settings
3. Experiment with different LoRa presets to see range differences
4. Compare 30m vs 90m resolution outputs for your use case
