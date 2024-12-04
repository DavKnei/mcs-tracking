# MCS Detection and Tracking

This repository contains Python code for detecting and tracking **Mesoscale Convective Systems (MCS)** based on gridded precipitation data. 
The code processes precipitation data from NetCDF files, identifies MCS regions, and tracks them over time using spatial overlap criteria.

## Features

- **Detection of MCS Regions**: Identifies moderate and heavy precipitation areas using configurable thresholds.
- **Clustering**: Uses clustering to group moderate precipitation grid points into potential MCS regions.
- **Convective Plumns**: Uses watershed segmentation to look for heavy precipitation regions in the clusters.
- **Cluster Filter**: Filter the moderate precipitation clusters by size and number of convective plumns.
- **Tracking Over Time**: Tracks MCS regions across multiple time steps based on spatial overlap.
- **Parallel Processing**: Supports parallel processing for efficient computation.
- **Visualization**: Generates plots of detected MCS regions for each time step.

## Table of Contents

- [Usage](#usage)
- [Code Structure](#code-structure)
- [Parameters and Configuration](#parameters-and-configuration)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Usage

1. **Prepare your data**:
   - Place your NetCDF files containing precipitation data in a directory.
   - Ensure that the NetCDF files have the variables `lat`, `lon`, `pr` (precipitation), and `time`.

2. **Configure parameters**:
   - Edit the `main.py` script to set the paths to your data directory and output directory.
   - Adjust detection and tracking parameters as needed (see [Parameters and Configuration](#parameters-and-configuration)).

3. **Run the main script**:

   ```bash
   python main.py
   ```

   This will perform MCS detection and tracking on your data and generate output files.

4. **View the results**:
   - Detected MCS regions and tracking information are saved in `mcs_detection_tracking_output.nc`.
   - Plots of detected MCS regions for each time step are saved in the specified output directory.

## Code Structure

- `main.py`: Main script that orchestrates the detection and tracking workflow.
- `detection_hdbscan.py`: Contains functions for data loading, preprocessing, and MCS detection ussing HDBSCAN for clustering.
- `tracking.py`: Implements the tracking of MCS regions across time steps.
- `plot.py`: Includes functions for visualizing precipitation data and detected MCS regions.
- `requirements.txt`: Lists the required Python packages.

## Parameters and Configuration

### In `main.py`:

- **Data Directories**:

  ```python
  data_directory = '/path/to/your/data/'       # Directory containing NetCDF files
  output_plot_dir = '/path/to/save/plots/'     # Directory to save the plots
  ```

- **Number of Cores for Parallel Processing**:

  ```python
  NUMBER_OF_CORES = 4  # Set this to the number of cores you want to use
  ```
  ```python
  USE_MULTIPROCESSING = True  # Disable multiprocessing for debugging
  ```
### Detection Parameters in `detection.py`:

- **Precipitation Thresholds**:

  ```python
  moderate_prec_threshold = 1    # Moderate precipitation threshold (mm)
  heavy_prec_threshold = 15      # Heavy precipitation threshold (mm)
  ```

- **Clustering Parameters**:

  ```python
  min_cluster_size = 50  # HDBSCAN  Minimum size of clusters
  ```

- **Area Threshold**:

  ```python
  min_area_km2 = 10000  # Minimum cluster area in square kilometers
  ```
- **Convective plumns threshold**:

  ```python
    min_plumes = 2  # minimum number of heavy precipitation regions inside a cluster
  ```

- **Grid Spacing**:

  ```python
  grid_spacing = 4  # Grid spacing in kilometers
  ```

### Tracking Parameters in `tracking.py`:

- **Minimum Overlap Percentage**:

  ```python
  min_overlap_percentage = 10  # Minimum percentage overlap for tracking
  ```

### Visualization Parameters in `plot.py`:

- **Map Extent**:

  ```python
  ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
  ```

  Adjust `lon_min`, `lon_max`, `lat_min`, and `lat_max` to focus on your area of interest.

- **Minimum Precipitation Threshold for Plotting**:

  ```python
  min_prec_threshold = 0.1  # Minimum precipitation to display in plots (mm)
  ```

## Example

An example workflow to run the MCS detection and tracking:

1. **Set up the data directory**:
   - Place your NetCDF precipitation data files in `data/`.

2. **Edit `main.py`**:
   - Set `data_directory` to `data/`.
   - Set `output_plot_dir` to `plots/`.

3. **Run the script**:

   ```bash
   python main.py
   ```

4. **Output**:
   - Detected MCS regions and tracking information are saved in `mcs_detection_tracking_output.nc`.
   - Plots are saved in the `plots/` directory.

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.


