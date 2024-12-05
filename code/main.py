# main.py

import os
import glob
import concurrent.futures
import numpy as np
import xarray as xr
from detection import detect_mcs_in_file
from tracking import track_mcs, filter_main_mcs
from plot import save_detection_plot, save_intermediate_plots
from input_output import save_detection_results, load_detection_results, save_tracking_results_to_netcdf

# Define a function for parallel processing
def process_file(file_path):
    result = detect_mcs_in_file(file_path)
    return result

def main():
    # Directory containing NetCDF files
    data_directory = "/nas/home/dkn/Desktop/PyFLEXTRKR_WRF_ref/WRF_test/WRF_test_data/wrf_rainrate_processed/"

    output_path = "/nas/home/dkn/Desktop/MCS-tracking/output_data/wrf_test/"
    output_plot_dir = "/nas/home/dkn/Desktop/MCS-tracking/output_data/wrf_test/figures/hdbscan"
    tracking_output_dir = "/nas/home/dkn/Desktop/MCS-tracking/output_data/wrf_test/tracking_results/"

    detection_results_file = os.path.join(output_path, 'detection_results.nc')

    # List all NetCDF files in the directory
    file_list = sorted(glob.glob(os.path.join(data_directory, '*.nc')))

    # List to hold detection results
    detection_results = []

    USE_MULTIPROCESSING = True
    SAVE_PLOTS = False

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
            # Specify the number of cores
            NUMBER_OF_CORES = 24  # Adjust based on your system

            # Use ProcessPoolExecutor for CPU-bound tasks
            with concurrent.futures.ProcessPoolExecutor(max_workers=NUMBER_OF_CORES) as executor:
                # Map the files to the process_file function
                futures = [executor.submit(process_file, file_path) for file_path in file_list]
                for future in concurrent.futures.as_completed(futures):
                    detection_result = future.result()
                    detection_results.append(detection_result)
                    print(f"MCS detection completed for time {detection_result['time']}.")
        else:
            # Process files sequentially
            for file_path in file_list:
                detection_result = detect_mcs_in_file(file_path)
                detection_results.append(detection_result)
                print(f"MCS detection completed for {file_path}.")
                if SAVE_PLOTS:
                    save_intermediate_plots(detection_result, output_plot_dir)

        # Sort detection results by time to ensure correct sequence
        detection_results.sort(key=lambda x: x['time'])
        print('Detection finished.')

        # Save detection results to NetCDF file
        save_detection_results(detection_results, detection_results_file)
    else:
        # Detection results were loaded from file
        pass

    # Now, generate and save plots before tracking
    if SAVE_PLOTS:
        for detection_result in detection_results:
            final_labeled_regions = detection_result['final_labeled_regions']
            prec = detection_result['precipitation']  # Ensure 'precipitation' is in detection_result
            lat = detection_result['lat']
            lon = detection_result['lon']
            file_time = detection_result['time']
            file_time_str = np.datetime_as_string(file_time, unit='h')  # Convert time to string format

            save_detection_plot(
                lon=lon,
                lat=lat,
                prec=prec,
                final_labeled_regions=final_labeled_regions,
                file_time=file_time_str,
                output_dir=output_plot_dir,
                min_prec_threshold=0.1  # Minimum precipitation to plot in color
            )

    # Perform tracking
    print('Tracking of MCS...')
    mcs_detected_list, mcs_id_list, lifetime_list, time_list, lat, lon, main_mcs_ids = track_mcs(detection_results, main_lifetime_thresh=6)
    print('Tracking of MCS finished.')

    # Optionally filter to main MCS
    mcs_id_list_filtered = filter_main_mcs(mcs_id_list, main_mcs_ids)  # Optional maybe deactivate

    # Save tracking results (use filtered lists if desired)
    save_tracking_results_to_netcdf(mcs_detected_list, mcs_id_list_filtered, lifetime_list, time_list, lat, lon, tracking_output_dir)


if __name__ == "__main__":
    main()
