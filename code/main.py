import os
import glob
import concurrent.futures
import numpy as np
import xarray as xr
from detection import detect_mcs_in_file
from tracking import track_mcs
from plot import save_detection_plot

# Define a function for parallel processing -> put it outside main() to make it a global function which is picklable
def process_file(file_path):
    result = detect_mcs_in_file(file_path)
    return result

def main():
    # Directory containing NetCDF files
    data_directory = "/nas/home/dkn/Desktop/PyFLEXTRKR_WRF_ref/WRF_test/WRF_test_data/wrf_rainrate_processed/"

    output_path = "/nas/home/dkn/Desktop/MCS-tracking/output_data/wrf_test/"
    output_plot_dir = "/nas/home/dkn/Desktop/MCS-tracking/output_data/wrf_test/figures"

    # List all NetCDF files in the directory
    file_list = sorted(glob.glob(os.path.join(data_directory, '*.nc')))

    # List to hold detection results
    detection_results = []

    # Specify the number of cores
    NUMBER_OF_CORES = 24  # 40 available on wegc_comp


    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUMBER_OF_CORES) as executor:
        # Map the files to the process_file function
        futures = [executor.submit(process_file, file_path) for file_path in file_list]
        for future in concurrent.futures.as_completed(futures):
            print(f'MCS detection in {process_file}...')
            detection_results.append(future.result())
            

    # Sort detection results by time to ensure correct sequence
    detection_results.sort(key=lambda x: x['time'])
    print('Detection of finsihed')

    # Now, generate and save plots before tracking
    for detection_result in detection_results:
        final_labeled_regions = detection_result['final_labeled_regions']
        prec = detection_result['prec']
        lat = detection_result['lat']
        lon = detection_result['lon']
        file_time = detection_result['time']
        file_time_str = np.datetime_as_string(file_time, unit='h')  # Convert time to string format

        # Call the plotting function
        save_detection_plot(
            lon=lon,
            lat=lat,
            prec=prec,
            final_labeled_regions=final_labeled_regions,
            file_time=file_time_str,
            output_dir=output_plot_dir,
            min_prec_threshold=0.1  # Adjust if necessary
        )

    # Perform tracking
    print('Tracking of MCS...')
    mcs_detected_list, mcs_id_list, time_list, lat, lon = track_mcs(detection_results)
    print('Tracking of MCS finsihed')
    # Stack the results into arrays
    mcs_detected_all = np.stack(mcs_detected_list, axis=0)
    mcs_id_all = np.stack(mcs_id_list, axis=0)

    # Create the dataset
    output_ds = xr.Dataset(
        {
            'mcs_detected': (['time', 'y', 'x'], mcs_detected_all),
            'mcs_id': (['time', 'y', 'x'], mcs_id_all)
        },
        coords={
            'time': time_list,
            'lat': (['y', 'x'], lat.values),
            'lon': (['y', 'x'], lon.values)
        },
        attrs={
            'description': 'MCS detection and tracking results',
            'min_overlap_percentage': 10,
            'note': 'Merging and splitting events are not fully handled and may result in warnings.'
        }
    )

    print(f'Save file to {output_path}')
    # Save to NetCDF
    output_ds.to_netcdf(os.path.join(output_path, 'wrf_test_track.nc'))

if __name__ == "__main__":
    main()
