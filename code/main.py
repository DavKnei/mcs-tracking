# main.py

import os
import glob
import concurrent.futures
import argparse
import yaml
import sys
import re
import logging
import pandas as pd
from collections import defaultdict

from detection_main import detect_mcs_in_file
from tracking_main import track_mcs
from tracking_filter_func import filter_main_mcs
from input_output import (
    save_detection_result,
    load_individual_detection_files,
    save_tracking_result,
)
from logging_setup import setup_logging, handle_exception

# Set a global exception handler to log uncaught exceptions
sys.excepthook = handle_exception


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
    lifting_index_percentage,
    grid_spacing_km,
):
    """
    Wrapper function to run MCS detection for a single file.
    This is used as the target for parallel processing.
    """
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
        lifting_index_percentage,
        grid_spacing_km,
    )
    return result


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MCS Detection and Tracking")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration YAML file."
    )
    return parser.parse_args()


def group_files_by_year(file_list):
    """
    Groups a list of file paths into a dictionary keyed by year.

    This function uses a regular expression to find a date in YYYYMMDD format
    within the filename, making it robust to different naming conventions.
    """
    files_by_year = defaultdict(list)
    # This regex looks for a sequence of 8 digits (YYYYMMDD)
    date_pattern = re.compile(r"(\d{8})")

    for f in file_list:
        basename = os.path.basename(f)
        match = date_pattern.search(basename)

        if match:
            date_str = match.group(1)
            try:
                # Convert the extracted 8-digit string to a datetime object
                year = pd.to_datetime(date_str, format="%Y%m%d").year
                files_by_year[year].append(f)
            except ValueError:
                logging.warning(
                    f"Found potential date '{date_str}' in '{basename}', but it's not a valid date. Skipping."
                )
        else:
            logging.warning(
                f"Could not parse YYYYMMDD date from filename: {basename}. Skipping."
            )

    return files_by_year


def main():
    """
    Main execution script for MCS detection and tracking.

    The workflow is structured as follows:
    1. Load configuration from a YAML file.
    2. Group input data files by year.
    3. Loop through each year for processing:
        a. Run MCS detection for every timestep, saving results to hourly files.
        b. Load the year's detection results back into memory.
        c. Run the tracking algorithm on the full year's data.
        d. Save the final tracking results to hourly files.
    This yearly batch approach ensures scalability for multi-year datasets.
    """
    # --- 1. SETUP AND CONFIGURATION ---
    logger = logging.getLogger(__name__)
    args = parse_arguments()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # General parameters
    precip_data_dir = config["precip_data_directory"]
    file_suffix = config["file_suffix"]
    detection_output_path = config["detection_output_path"]
    tracking_output_dir = config["tracking_output_dir"]
    grid_spacing_km = config["grid_size_km"]
    precip_data_var = config["precip_var_name"]
    lat_name = config["lat_name"]
    lon_name = config["lon_name"]
    data_source = config["data_source"]

    # Detection parameters
    min_size_threshold = config["min_size_threshold"]
    heavy_precip_threshold = config["heavy_precip_threshold"]
    moderate_precip_threshold = config["moderate_precip_threshold"]
    min_nr_plumes = config["min_nr_plumes"]
    lifting_index_percentage = config["lifting_index_percentage_threshold"]

    # Tracking parameters
    main_lifetime_thresh = config["main_lifetime_thresh"]
    main_area_thresh = config["main_area_thresh"]
    grid_cell_area_km2 = config["grid_cell_area_km2"]
    nmaxmerge = config["nmaxmerge"]

    # Operational parameters
    USE_MULTIPROCESSING = config["use_multiprocessing"]
    NUMBER_OF_CORES = config["number_of_cores"]
    DO_DETECTION = config["detection"]
    USE_LIFTING_INDEX = config["use_lifting_index"]

    # Setup logging and create output directories
    os.makedirs(detection_output_path, exist_ok=True)
    os.makedirs(tracking_output_dir, exist_ok=True)
    setup_logging(detection_output_path)
    logger.info("Configuration loaded and logging initialized.")

    # --- 2. GROUP INPUT FILES BY YEAR ---
    all_precip_files = sorted(
        glob.glob(os.path.join(precip_data_dir, f"*{file_suffix}"))
    )
    if not all_precip_files:
        raise FileNotFoundError("Precipitation data directory is empty. Exiting.")

    files_by_year = group_files_by_year(all_precip_files)

    if USE_LIFTING_INDEX:
        lifting_index_data_dir = config["lifting_index_data_directory"]
        lifting_index_data_var = config["liting_index_var_name"]
        all_li_files = sorted(
            glob.glob(os.path.join(lifting_index_data_dir, f"*{file_suffix}"))
        )
        if not all_li_files:
            raise FileNotFoundError("Lifting index data directory is empty. Exiting.")
        li_files_by_year = group_files_by_year(all_li_files)

    # --- 3. MAIN YEARLY PROCESSING LOOP ---
    for year in sorted(files_by_year.keys()):
        logger.info(f"--- Starting processing for year: {year} ---")
        precip_file_list_year = files_by_year[year]

        if USE_LIFTING_INDEX:
            li_files_year = li_files_by_year.get(year, [])
            if len(precip_file_list_year) != len(li_files_year):
                logger.warning(
                    f"Mismatch in file counts for {year}. Precip: {len(precip_file_list_year)}, LI: {len(li_files_year)}. Skipping year."
                )
                continue
        else:
            li_files_year = [None] * len(precip_file_list_year)
            lifting_index_data_var = None

        # --- 3a. DETECTION PHASE ---
        if DO_DETECTION:
            logger.info(
                f"Running detection for {len(precip_file_list_year)} files in {year}..."
            )
            if USE_MULTIPROCESSING:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=NUMBER_OF_CORES
                ) as executor:
                    futures = [
                        executor.submit(
                            process_file,
                            precip_file,
                            precip_data_var,
                            li_file,
                            lifting_index_data_var,
                            lat_name,
                            lon_name,
                            heavy_precip_threshold,
                            moderate_precip_threshold,
                            min_size_threshold,
                            min_nr_plumes,
                            lifting_index_percentage,
                            grid_spacing_km,
                        )
                        for precip_file, li_file in zip(
                            precip_file_list_year, li_files_year
                        )
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            detection_result = future.result()
                            save_detection_result(
                                detection_result, detection_output_path, data_source
                            )
                        except Exception as e:
                            logger.error(f"A detection task failed: {e}")
            else:
                for precip_file, li_file in zip(precip_file_list_year, li_files_year):
                    detection_result = detect_mcs_in_file(
                        precip_file,
                        precip_data_var,
                        li_file,
                        lifting_index_data_var,
                        lat_name,
                        lon_name,
                        heavy_precip_threshold,
                        moderate_precip_threshold,
                        min_size_threshold,
                        min_nr_plumes,
                        lifting_index_percentage,
                        grid_spacing_km,
                    )
                    save_detection_result(
                        detection_result, detection_output_path, data_source
                    )
            logger.info(f"Detection for year {year} finished.")

        # --- 3b. LOADING PHASE ---
        logger.info(f"Loading all detection files for year {year}...")
        year_detection_dir = os.path.join(detection_output_path, str(year))
        detection_results = load_individual_detection_files(
            year_detection_dir, USE_LIFTING_INDEX
        )

        if not detection_results:
            logger.warning(
                f"No detection results found for year {year}. Skipping tracking."
            )
            continue

        # --- 3c. TRACKING PHASE ---
        logger.info(f"Starting tracking for year {year}...")
        (
            robust_mcs_id,          
            mcs_id,            
            mcs_id_merge_split,
            lifetime_list,
            time_list, lat, lon, merging_events, splitting_events, tracking_centers_list
        ) = track_mcs(
            detection_results, main_lifetime_thresh, main_area_thresh,
            grid_cell_area_km2, nmaxmerge, use_li_filter=USE_LIFTING_INDEX
        )
        logger.info(f"Tracking for year {year} finished.")


        # --- 3d. SAVING TRACKING PHASE ---
        logger.info(f"Saving individual hourly tracking files for year {year}...")
        for i in range(len(time_list)):
            # Package all data for this single timestep into a dictionary
            tracking_data_for_timestep = {
                "robust_mcs_id": robust_mcs_id[i],         
                "mcs_id": mcs_id[i],           
                "mcs_id_merge_split": mcs_id_merge_split[i], 
                "lifetime": lifetime_list[i],
                "time": time_list[i],
                "lat": lat,
                "lon": lon,
                "tracking_centers": tracking_centers_list[i],
            }
            save_tracking_result(tracking_data_for_timestep, tracking_output_dir, data_source)


        logger.info(f"--- Finished processing for year: {year} ---")

    logger.info("All processing completed successfully.")


if __name__ == "__main__":
    main()
