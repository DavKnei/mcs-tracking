# MCS Detection and Tracking

This repository contains a Python-based framework for detecting and tracking **Mesoscale Convective Systems (MCSs)** using gridded precipitation and atmospheric instability data. The algorithm processes NetCDF files, identifies MCS candidates, and tracks them over time using a robust methodology that accounts for spatial overlap, lifetime, area, and the convective environment.

## Features

- **Multi-Stage Detection**: Identifies MCS candidates through a series of steps:
  - **Convective Cores**: Detects initial heavy precipitation cores using a `heavy_precip_threshold`.
  - **Cluster Growth**: Expands these cores into larger moderate precipitation regions using morphological dilation.
  - **Size Filtering**: Filters the resulting clusters by size to identify systems large enough to be considered potential MCSs.
- **Convective Environment Flagging**: Evaluates the atmospheric stability for each detected cluster. If a specified percentage of a cluster's area (e.g., >20%) has a Lifting Index below a threshold (e.g., -2 K used in [Future Changes in European Severe Convection Environments in a Regional Climate Model Ensemble](https://journals.ametsoc.org/doi/10.1175/JCLI-D-16-0777.1)), the entire cluster is flagged as being in a "convective environment" for that timestep.
- **Advanced Tracking**: Tracks systems across multiple time steps, correctly handling complex **merging and splitting** events to maintain a coherent history of each system.
- **Robust MCS Identification**: A tracked system is confirmed as a "main MCS" only if it undergoes a continuous period of maturity, defined by meeting both the **area and lifetime thresholds** while also being in a **convective environment**.
- **Parallel Processing**: Leverages multiprocessing for the computationally intensive detection step, significantly speeding up the workflow.
- **Scalable Architecture**: The processing is handled in yearly batches, allowing the algorithm to analyze decades of data without excessive memory consumption.

---
## Table of Contents

- [Usage](#usage)
- [Configuration](#configuration)
- [Code Structure](#code-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---
## Usage

### 1. Prepare Your Data
- Place your gridded NetCDF precipitation files in a dedicated directory.
- If using the convective environment filter, place your corresponding Lifting Index NetCDF files in their own directory. The data should have the same spatial and temporal resolution.
- Ensure your NetCDF files contain latitude, longitude, and time variables.

### 2. Set Configuration
- Create a `config.yaml` file to define all parameters, including data paths, variable names, and the scientific thresholds for detection and tracking.

### 3. Run the Main Script
- Execute the workflow from your terminal. Use the `--config` argument to point to your configuration file:
   ```bash
   python main.py --config config.yaml
   ```

### 4. View the Results
The script produces hourly NetCDF files for both detection and tracking, neatly organized into `YYYY/MM/` subdirectories. The primary tracking output files (e.g., `tracking_YYYYMMDDTHHMMSS.nc`) contain three key variables that describe the MCSs at different levels of detail:

- **`robust_mcs_id`**: The most restrictive filter. It shows the pixels of a main MCS **only** during the timesteps where the system is **simultaneously** larger than `main_area_thresh` and meets the convective LI criteria. This isolates the mature, "in-phase" portion of the MCSs.
- **`main_mcs_id`**: Shows the **full lifetime** of all systems that were successfully identified as main MCSs. This allows you to see the complete formation and decay phases of significant storms.
- **`main_mcs_id_merge_split`**: The most inclusive filter. This shows the **full "family tree"** of a main MCS, including the complete tracks of any smaller systems that merged into it or split from it.
- **`lifetime`**: A pixel-wise record of the lifetime (in timesteps) of any tracked cloud cluster.

---
## Configuration

All parameters are controlled via a `config.yaml` file. Below are examples of the key sections.

- **Data and Paths**:
  ```yaml
  precip_data_directory: "./tests/Test/data/"
  lifting_index_data_directory: "./tests/Test/data/"
  file_suffix: ".nc_test"
  detection_output_path: "./tests/Test/output/detection/"
  tracking_output_dir: "./tests/Test/output/tracking/"

  grid_size_km: 4.0
  precip_var_name: "pr"
  lifting_index_var_name: "li"
  lat_name: "lat"
  lon_name: "lon"
  data_source: "Test data"
  ```

- **Detection Thresholds**:
  ```yaml
  min_size_threshold: 10
  heavy_precip_threshold: 10.0
  moderate_precip_threshold: 1.0
  min_nr_plumes: 1
  lifting_index_percentage_threshold: 0.2
  ```

- **Tracking Thresholds**:
  ```yaml
  main_lifetime_thresh: 5
  main_area_thresh: 5000
  grid_cell_area_km2: 16
  nmaxmerge: 5
  ```
- **Operational Parameters**:
  ```yaml
  use_lifting_index: True
  detection: True
  use_multiprocessing: False
  number_of_cores: 24
  ```

---
## Code Structure

- `main.py`: The main executable script that orchestrates the entire yearly batch processing workflow.
- `config.yaml`: Configuration file for all user-defined parameters.
- `detection_main.py`: Contains the high-level function for detecting MCS candidates in a single data file.
- `detection_filter_func.py`: Includes functions for filtering detected clusters by size and convective environment (Lifting Index).
- `tracking_main.py`: Implements the core logic for tracking systems over time and applying the final robust MCS filtering.
- `tracking_merging.py` / `tracking_splitting.py`: Contain specific logic to handle merging and splitting events.
- `tracking_filter_func.py`: Contains functions to filter the final tracking results into the different output variables (e.g., `main_mcs_id`, `main_mcs_id_merge_split`).
- `input_output.py`: Manages all file I/O, including loading input data and saving the hourly detection and tracking results.
- `tests/`: Directory containing test cases and sample data to verify the algorithm's correctness, particularly for merging, splitting, and LI filtering scenarios.

---
## Testing

A test case is located in the `tests/Test` directory, complete with sample data and a pre-made configuration file. It is designed to validate the algorithm's behavior in complex scenarios, including merges, splits, and varied convective environments. The test data uses a `.nc_test` suffix to distinguish it from operational data. While the test runs automatically, the final output should be visually inspected to confirm expected behavior.

---
## Contributing

Contributions are welcome! If you have ideas for improvements or new features, please open an issue or submit a pull request.

---
## License

This project is licensed under the MIT License.
