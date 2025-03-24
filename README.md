# MCS Detection and Tracking

This repository contains Python code for detecting and tracking **Mesoscale Convective Systems (MCS)** based on gridded precipitation data. 
The code processes precipitation data from NetCDF files, identifies MCS regions and smaller convective features, and tracks them over time using spatial overlap criteria and clustering-based approaches.

## Features

- **Detection of MCS Regions**: Identifies moderate and heavy precipitation areas using configurable thresholds.
- **Convective Cores**: Uses clustering based connected components to find heavy precipitation cores above heavy_precip_threshold.
- **MCS Canditate Clusters**: Use morphological dilation to expand the convective cores outward. If two cluster meet, they get merged immediately.
- **Cluster Filter**: Filter the moderate precipitation clusters by size according to min_size_threshold for MCS candidates.
- **Lifting Index Flag**: Clusters with a lifting index smaller than -2 K for over 50% of their area get a positive lifting index flag (convective environment).
- **Tracking Over Time**: Tracks MCS regions across multiple time steps based on spatial overlap.
- **Merging and Splitting Events**: Handles complex scenarios where MCSs merge into a single larger system or split into multiple smaller systems.
- **Lifting Index Filter**: Systems that dont have a positive lifting index flag over the whole liftetime get removed. 
- **Parallel Processing**: Supports parallel processing for the detection step for efficient computation.

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
   - Place your NetCDF files containing precipitation data in a directory if you want to include the lifting index criteria as well, make sure lifting index data is provided.
   - Ensure that the NetCDF files have the variables `lat`, `lon`, (precipitation), and `time`.
   - The precipitation data should have at least hourly resolution. The lifting index data should also be provided in the same temporal and spatial resolution.

2. **Set configuration**:
   - Create or edit the `config.yaml` file to set the paths to your data directory and thresholds. For example,
  - Data directories
  - Variable names
  - Detection thresholds (see [Parameters and Configuration](#parameters-and-configuration))
  - Tracking thresholds (see [Parameters and Configuration](#parameters-and-configuration))
  - Additional flags for parallel computing, number of computation cores, ...

3. **Run the main script**:
  Use the `--config`argument to specify the configuration file:
   ```bash
   python main.py --config config.yaml
   ```

   This will perform MCS detection and tracking on your data and generate output files according to the config file.

4. **View the results**:
   - Detected MCS regions and tracking information are saved the output files. E.g. `detection_results.nc`, `mcs_detection_tracking_output.nc`.

## Code Structure

- `main.py`: Main script that orchestrates the detection and tracking workflow.
- `detection.py`: Contains functions for data loading, preprocessing, and MCS detection.
- `tracking.py`: Implements the tracking of MCS regions across time steps.
- `plot.py`: Includes functions for visualizing precipitation data and detected MCS regions.
- `tests/`: Directory containing test cases. Currently a single test case including merging and splitting scenarios and lifting index criteria.
- `requirements`: Lists the required Python packages.

## Parameters and Configuration

### In `main.py`:

- **Data Directories**:

### Example:
  ```python
precip_data_directory: "./tests/Test/data/"
lifting_index_data_directory: "./tests/Test/data/"
file_suffix: ".nc_test"
detection_output_path: "./tests/Test/"
tracking_output_dir: "./tests/Test/"

grid_size_km: 4.0
precip_var_name: "pr"
liting_index_var_name: "li"
lat_name: "lat"
lon_name: "lon"
data_source: "Test data"
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
main_area_thresh: 5000
grid_cell_area_km2: 16
nmaxmerge: 5
```
- **Other Parameters**:
```python
use_lifting_index: True
detection: True
use_multiprocessing: False
number_of_cores: 24
```

## Testing

Currently their is a single test case located in `tests/Test`. The test is designed to test most possible cases: Merging, Splitting, non convective environments and so on. The test data use `.nc_test` suffix for their input files to avoid large file commits and to distinguish test data from actual operational data.


### What the Tests Do
By running these tests, you can confidently ensure that changes to the code maintain or improve algorithmic correctness and stability.

- The output still has to be visually inspected!

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.


