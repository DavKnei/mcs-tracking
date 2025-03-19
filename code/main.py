import os
import glob
import concurrent.futures
import argparse
import yaml
import sys
import logging

from detection_main import detect_mcs_in_file
from tracking_main import track_mcs
from tracking_filter_func import filter_main_mcs
from input_output import (
    save_detection_results,
    load_detection_results,
    save_tracking_results_to_netcdf,
)
from logging_setup import setup_logging, handle_exception

sys.excepthook = handle_exception


# Define a function for parallel processing
def process_file(
    precip_file_path,
    precip_data_var,
    lifting_index_file_path,
    lifting_index_data_var,
    lat_name,
    lon_name,
    heavy_precip_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    grid_spacing_km,
):
    result = detect_mcs_in_file(
        precip_file_path,
        precip_data_var,
        lifting_index_file_path,
        lifting_index_data_var,
        lat_name,
        lon_name,
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

    # Get a logger for this module
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    args = parse_arguments()

    # Load the configuration file
    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Access parameters from config
    precip_data_dir = config["precip_data_directory"]
    file_suffix = config["file_suffix"]
    output_path = config["output_path"]
    tracking_output_dir = config["tracking_output_dir"]
    grid_spacing_km = config["grid_size_km"]
    precip_data_var = config["precip_var_name"]
    lat_name = config["lat_name"]
    lon_name = config["lon_name"]
    data_source = config["data_source"]

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
    USE_MULTIPROCESSING = config.get("use_multiprocessing", True)
    NUMBER_OF_CORES = config.get("number_of_cores", 24)
    DO_DETECTION = config.get("detection", True)

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    setup_logging(output_path)
    logger.info("Loading Configuration finished.")

    # Ensure directories exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tracking_output_dir, exist_ok=True)

    detection_results_file = os.path.join(output_path, "detection_results.nc")

    # List all NetCDF files in the directory
    precip_file_list = sorted(
        glob.glob(os.path.join(precip_data_dir, f"*{file_suffix}"))
    )
    if not precip_file_list:
        raise FileNotFoundError(
            "File directory is empty or no files found matching the specified suffix. Exiting..."
        )

    # Other parameters: Lifting Index
    USE_LIFTING_INDEX = config.get("use_lifting_index", True)

    if USE_LIFTING_INDEX:
        lifting_index_data_dir = config["lifting_index_data_directory"]
        lifting_index_data_var = config["liting_index_var_name"]

        lifting_index_file_list = sorted(
            glob.glob(os.path.join(lifting_index_data_dir, f"*{file_suffix}"))
        )
        if not lifting_index_file_list:
            raise FileNotFoundError(
                "File directory is empty or no files found matching the specified suffix. Exiting..."
            )
    else:
        lifting_index_file_list = [None] * len(precip_file_list)
        lifting_index_data_var = None

    # List to hold detection results
    detection_results = []

    # Check if detection results file exists and is valid
    detection_results_exist = os.path.exists(detection_results_file)

    if detection_results_exist and not DO_DETECTION:
        detection_results = load_detection_results(detection_results_file)
        if detection_results is not None:
            logger.info("Detection results loaded from file. Skipping detection step.")
        else:
            logger.warning("Detection results file is invalid. Running detection.")
            detection_results_exist = False  # Set to False to run detection
    elif detection_results_exist and DO_DETECTION:
        detection_results_exist = False  # Set to False to run detection
        logger.info("Detection file does exist. detection=True: Running detection")
    else:
        logger.info("Detection results file does not exist. Running detection.")

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
                        precip_file_path,
                        precip_data_var,
                        lifting_index_file_path,
                        lifting_index_data_var,
                        lat_name,
                        lon_name,
                        heavy_precip_threshold,
                        moderate_precip_threshold,
                        min_size_threshold,
                        min_nr_plumes,
                        grid_spacing_km,
                    )
                    for precip_file_path, lifting_index_file_path in zip(
                        precip_file_list, lifting_index_file_list
                    )
                ]
                for future in concurrent.futures.as_completed(futures):
                    detection_result = future.result()
                    detection_results.append(detection_result)
        else:
            # Process files sequentially
            for precip_file_path, lifting_index_file_path in zip(
                precip_file_list, lifting_index_file_list
            ):
                detection_result = detect_mcs_in_file(
                    precip_file_path,
                    precip_data_var,
                    lifting_index_file_path,
                    lifting_index_data_var,
                    lat_name,
                    lon_name,
                    heavy_precip_threshold,
                    moderate_precip_threshold,
                    min_size_threshold,
                    min_nr_plumes,
                    grid_spacing_km,
                )
                detection_results.append(detection_result)

        # Sort detection results by time to ensure correct sequence
        detection_results.sort(key=lambda x: x["time"])
        logger.info("Detection finished.")
        # Save detection results to NetCDF file
        save_detection_results(detection_results, detection_results_file, data_source)
    else:
        # Detection results were loaded from file
        pass

    # Perform tracking
    logger.info("Tracking of MCS...")
    (
        mcs_ids_list,
        main_mcs_ids,
        lifetime_list,
        time_list,
        lat,
        lon,
        merging_events,
        splitting_events,
        tracking_centers_list,
    ) = track_mcs(
        detection_results,
        main_lifetime_thresh,
        main_area_thresh,
        grid_cell_area_km2,
        nmaxmerge,
    )
    logger.info("Tracking of MCS finished.")

    main_mcs_ids_list = filter_main_mcs(mcs_ids_list, main_mcs_ids)

    # Save tracking results (use filtered lists if desired)
    save_tracking_results_to_netcdf(
        mcs_ids_list,
        main_mcs_ids_list,
        lifetime_list,
        time_list,
        lat,
        lon,
        tracking_centers_list,
        tracking_output_dir,
        data_source,
    )

    print("Tracking finished successfully.")
    logger.info("Tracking finished successfully.")
    logger.info("All processing completed.")


if __name__ == "__main__":
    main()
