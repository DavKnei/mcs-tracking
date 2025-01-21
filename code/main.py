# main.py

import os
import glob
import concurrent.futures
import argparse
import numpy as np
import xarray as xr
import yaml
from detection import detect_mcs_in_file
from tracking import track_mcs, filter_main_mcs
from plot import save_intermediate_plots
from input_output import (
    save_detection_results,
    load_detection_results,
    save_tracking_results_to_netcdf,
)

# Define a function for parallel processing
def process_file(
    file_path,
    data_var,
    heavy_precip_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    grid_spacing_km,
):
    result = detect_mcs_in_file(
        file_path,
        data_var,
        heavy_precip_threshold,
        moderate_precip_threshold,
        min_size_threshold,
        min_nr_plumes,
        grid_spacing_km,
    )
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description="MCS Detection and Tracking")
    parser.add_argument(
        "--config", type=str, help="Path to configuration file", required=True
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load the configuration file
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Access parameters from config
    data_directory = config["data_directory"]
    file_suffix = config["file_suffix"]
    output_path = config["output_path"]
    output_plot_dir = config["output_plot_dir"]
    tracking_output_dir = config["tracking_output_dir"]
    grid_spacing_km = config["grid_size_km"]
    data_var = config["var_name"]

    # Detection parameters
    min_size_threshold = config.get("min_size_threshold", 10)
    heavy_precip_threshold = config.get("heavy_precip_threshold", 10.0)
    moderate_precip_threshold = config.get("moderate_precip_threshold", 1.0)
    min_nr_plumes = config.get("min_nr_plumes", 1)

    # Tracking parameters
    main_lifetime_thresh = config.get("main_lifetime_thresh", 6)
    main_area_thresh = config.get("main_area_thresh", 10000)
    grid_cell_area_km2 = config.get("grid_cell_area_km2", 16)
    nmaxmerge = config.get("nmaxmerge", 5)

    # Other parameters
    SAVE_PLOTS = config.get("plotting_enabled", True)
    USE_MULTIPROCESSING = config.get("use_multiprocessing", True)
    NUMBER_OF_CORES = config.get("number_of_cores", 24)

    # Ensure directories exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(tracking_output_dir, exist_ok=True)

    detection_results_file = os.path.join(output_path, "detection_results.nc")

    # List all NetCDF files in the directory
    file_list = sorted(glob.glob(os.path.join(data_directory, f"*{file_suffix}")))
    if not file_list:
        raise FileNotFoundError("File directory is empty or no files found matching the specified suffix. Exiting...")

    # List to hold detection results
    detection_results = []

    # Check if detection results file exists and is valid
    detection_results_exist = os.path.exists(detection_results_file)

    if detection_results_exist:
        detection_results = load_detection_results(detection_results_file)
        if detection_results is not None:
            print("Detection results loaded from file. Skipping detection step.")
        else:
            print("Detection results file is invalid. Running detection.")
            detection_results_exist = False  # Set to False to run detection
    else:
        print("Detection results file does not exist. Running detection.")

    if not detection_results_exist:
        if USE_MULTIPROCESSING:

            # Use ProcessPoolExecutor for CPU-bound tasks
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=NUMBER_OF_CORES
            ) as executor:
                # Map the files to the process_file function
                futures = [
                    executor.submit(
                        process_file,
                        file_path,
                        data_var,
                        heavy_precip_threshold,
                        moderate_precip_threshold,
                        min_size_threshold,
                        min_nr_plumes,
                        grid_spacing_km,
                    )
                    for file_path in file_list
                ]
                for future in concurrent.futures.as_completed(futures):
                    detection_result = future.result()
                    detection_results.append(detection_result)
                    print(
                        f"MCS detection completed for time {detection_result['time']}."
                    )
        else:
            # Process files sequentially
            for file_path in file_list:
                detection_result = detect_mcs_in_file(
                    file_path,
                    data_var,
                    heavy_precip_threshold,
                    moderate_precip_threshold,
                    min_size_threshold,
                    min_nr_plumes,
                    grid_spacing_km,
                )
                detection_results.append(detection_result)

                print(f"MCS detection completed for {file_path}.")
                if SAVE_PLOTS:
                    save_intermediate_plots(detection_result, output_plot_dir)

        # Sort detection results by time to ensure correct sequence
        detection_results.sort(key=lambda x: x["time"])
        print("Detection finished.")
        # Save detection results to NetCDF file
        save_detection_results(detection_results, detection_results_file)
    else:
        # Detection results were loaded from file
        pass

    # Perform tracking
    print("Tracking of MCS...")
    (
        mcs_detected_list,
        mcs_id_list,
        lifetime_list,
        time_list,
        lat,
        lon,
        main_mcs_ids,
        merging_events,
        splitting_events,
    ) = track_mcs(
        detection_results,
        main_lifetime_thresh,
        main_area_thresh,
        grid_cell_area_km2,
        nmaxmerge,
    )
    print("Tracking of MCS finished.")

    mcs_id_list_filtered = filter_main_mcs(mcs_id_list, main_mcs_ids)

    # Save tracking results (use filtered lists if desired)
    save_tracking_results_to_netcdf(
        mcs_detected_list,
        mcs_id_list_filtered,
        lifetime_list,
        time_list,
        lat,
        lon,
        tracking_output_dir,
    )


if __name__ == "__main__":
    main()
