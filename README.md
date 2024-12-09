# MCS Detection and Tracking

This repository contains Python code for detecting and tracking **Mesoscale Convective Systems (MCS)** based on gridded precipitation data. 
The code processes precipitation data from NetCDF files, identifies MCS regions and smaller convective features, and tracks them over time using spatial overlap criteria and clustering-based approaches.

## Features

- **Detection of MCS Regions**: Identifies moderate and heavy precipitation areas using configurable thresholds.
- **Clustering**: Uses clustering based on HDBSCAN to group moderate precipitation grid points into potential MCS regions.
- **Convective Plums**: Uses watershed segmentation to look for heavy precipitation regions in the clusters.
- **Cluster Filter**: Filter the moderate precipitation clusters by size and number of convective plumns.
- **Tracking Over Time**: Tracks MCS regions across multiple time steps based on spatial overlap.
- **Merging and Splitting Events**: Handles complex scenarios where MCSs merge into a single larger system or split into multiple smaller systems.
- **Parallel Processing**: Supports parallel processing for efficient computation.
- **Visualization**: Generates plots of detected MCS regions for each time step.

## Table of Contents

- [Usage](#usage)
- [Configuration](#config)
- [Code Structure](#code-structure)
- [Parameters and Configuration](#parameters-and-configuration)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Usage

1. **Prepare your data**:
   - Place your NetCDF files containing precipitation data in a directory.
   - Ensure that the NetCDF files have the variables `lat`, `lon`, `pr` (precipitation), and `time`.

2. **Set configuration**:
   - Create or edit the `config.yaml` file to set the paths to your data directory and thresholds. For example,
  - Data directories
  - Detection thresholds (see [Parameters and Configuration](#parameters-and-configuration))
  - Tracking thresholds (see [Parameters and Configuration](#parameters-and-configuration))
  - Plotting options

3. **Run the main script**:
  Use the `--config`argument to specify the configuration file:
   ```bash
   python main.py --config config.yaml
   ```

   This will perform MCS detection and tracking on your data and generate output files according to the config file.

4. **View the results**:
   - Detected MCS regions and tracking information are saved the output files. E.g. `detection_results.nc`, `mcs_detection_tracking_output.nc`.
   - Plots of detected MCS regions for each time step are saved in the specified output directory.

## Code Structure

- `main.py`: Main script that orchestrates the detection and tracking workflow.
- `detection.py`: Contains functions for data loading, preprocessing, and MCS detection using HDBSCAN for clustering.
- `tracking.py`: Implements the tracking of MCS regions across time steps.
- `plot.py`: Includes functions for visualizing precipitation data and detected MCS regions.
- `tests/`: Directory containing test cases (Test1 through Test6) and pytest scripts.
- `requirements`: Lists the required Python packages.

## Parameters and Configuration

### In `main.py`:

- **Data Directories**:

### Example:
  ```python
  data_directory: "./tests/Test1/data/"
  file_suffix: ".nc_test"
  output_path: "./tests/Test1/"
  output_plot_dir: "./tests/Test1/figures/"
  tracking_output_dir: "./tests/Test1/"
  grid_size_km: 4.0
  ```

- **Detection Thresholds**:
### Example:
```python
min_size_threshold: 10
heavy_precip_threshold: 10.0
moderate_precip_threshold: 1.0
min_nr_plumes: 1
```

- **Tracking Thresholds**:
### Example:
```python
main_lifetime_thresh: 6
main_area_thresh: 10000
grid_cell_area_km2: 16
nmaxmerge: 5
```
- **Other Parameters**:
```python
plotting_enabled: False
use_multiprocessing: False
```

## Testing

We have multiple test cases located in `tests/` (Test1 through Test6), each designed to test different scenarios (e.g., simple MCS detection, merging events, splitting events). Some tests use `.nc_test` suffix for their input files to avoid large file commits and to distinguish test data from actual operational data.

### Running Tests

From the project root directory, run the tests using:

```bash
pytest tests/
```

### What the Tests Do

- The tests run `main.py` directly with appropriate `config.yaml` files for each test scenario.
- They ensure `detection_results.nc` is recreated for every test run, guaranteeing that detection and tracking steps are performed fresh each time.
- For scenarios like merging and splitting (e.g., Test5, Test6), test data (`MCS-test5_...nc_test` and `MCS-test6_...nc_test` files) provide known evolving precipitation fields. The tests check that `main.py` runs without errors, that outputs are produced, and (optionally) that figures are generated.


By running these tests, you can confidently ensure that changes to the code maintain or improve algorithmic correctness and stability.

- The output still has to be visually inspected!

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.


