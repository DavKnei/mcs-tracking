import xarray as xr

def load_data(file_path, time_index=0):
    """
    Load the dataset and select the specified time step.
    
    Parameters:
    - file_path: Path to the NetCDF file.
    - time_index: Index of the time step to select.
    
    Returns:
    - ds: xarray Dataset for the selected time.
    - lat: 2D array of latitudes.
    - lon: 2D array of longitudes.
    - prec: 2D array of precipitation values.
    """
    ds = xr.open_dataset(file_path)
    ds = ds.isel(time=time_index)  # Select the specified time step
    lat = ds['lat'].values
    lon = ds['lon'].values
    prec = ds['pr']
    return ds, lat, lon, prec

def save_detection_results(detection_results, output_filepath):
    """
    Save detection results to a NetCDF file.

    Parameters:
    - detection_results: List of detection_result dictionaries.
    - output_filepath: Path to the output NetCDF file.
    """
    # Prepare data for saving
    times = []
    final_labeled_regions_list = []
    lat = None
    lon = None

    for detection_result in detection_results:
        times.append(detection_result['time'])
        final_labeled_regions_list.append(detection_result['final_labeled_regions'])
        if lat is None:
            lat = detection_result['lat']
            lon = detection_result['lon']

    # Stack the final_labeled_regions along a new time dimension
    final_labeled_regions_array = np.stack(final_labeled_regions_list, axis=0)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            'final_labeled_regions': (['time', 'y', 'x'], final_labeled_regions_array)
        },
        coords={
            'time': times,
            'lat': (['y', 'x'], lat),
            'lon': (['y', 'x'], lon)
        },
        attrs={
            'description': 'Detection results of MCSs',
            'note': 'This file contains the final labeled regions from MCS detection.'
        }
    )

    # Save to NetCDF file
    ds.to_netcdf(output_filepath)
    print(f"Detection results saved to {output_filepath}")

def load_detection_results(input_filepath):
    """
    Load detection results from a NetCDF file.

    Parameters:
    - input_filepath: Path to the input NetCDF file.

    Returns:
    - detection_results: List of detection_result dictionaries.
    """
    if not os.path.exists(input_filepath):
        print(f"File {input_filepath} does not exist.")
        return None

    try:
        ds = xr.open_dataset(input_filepath)
    except Exception as e:
        print(f"Error opening {input_filepath}: {e}")
        return None

    # Check if required variables are present
    required_vars = ['final_labeled_regions', 'lat', 'lon', 'time']
    for var in required_vars:
        if var not in ds.variables:
            print(f"Variable {var} not found in {input_filepath}.")
            return None

    # Extract data
    final_labeled_regions_array = ds['final_labeled_regions'].values
    times = ds['time'].values
    lat = ds['lat'].values
    lon = ds['lon'].values

    detection_results = []
    for idx, time in enumerate(times):
        detection_result = {
            'final_labeled_regions': final_labeled_regions_array[idx],
            'time': time,
            'lat': lat,
            'lon': lon
        }
        detection_results.append(detection_result)

    print(f"Detection results loaded from {input_filepath}")
    return detection_results
