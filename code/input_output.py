import xarray as xr
import numpy as np
import os
import datetime
import json


def load_data(file_path, data_var, time_index=0):
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
    ds["time"] = ds["time"].values.astype("datetime64[s]")
    latitude = ds["lat"].values
    longitude = ds["lon"].values

    # Make lat and lon 2d
    if latitude.ndim == 1 and longitude.ndim == 1:
        lat, lon = np.meshgrid(longitude, latitude)
    else:
        lat, lon = latitude, longitude

    prec = ds[str(data_var)]
    return ds, lat, lon, prec


def save_detection_results(detection_results, output_filepath):
    """Saves detection results (including per-timestep center_of_mass) to a NetCDF file.

    This function gathers:
      - final_labeled_regions (stacked along time)
      - lat and lon arrays
      - center_points (dictionary of label -> (lat, lon)) for each timestep,
        stored as a JSON attribute.

    Args:
        detection_results (List[dict]):
            Each dict must contain:
                "final_labeled_regions": 2D array of labeled clusters,
                "lat": 2D array of latitudes,
                "lon": 2D array of longitudes,
                "time": Timestamp or datetime-like,
                optionally "center_points": { label_value : (lat, lon) }.
        output_filepath (str):
            Path to the output NetCDF file.
    """
    times = []
    final_labeled_regions_list = []
    lat = None
    lon = None
    center_points_list = []  # Will store JSON-encoded center points

    for detection_result in detection_results:
        # Extract the required info
        times.append(detection_result["time"])
        final_labeled_regions_list.append(detection_result["final_labeled_regions"])

        if lat is None:
            lat = detection_result["lat"]
            lon = detection_result["lon"]

        # If 'center_points' is present, store it; else store empty
        if "center_points" in detection_result:
            center_points = detection_result["center_points"]
        else:
            center_points = {}

        # Convert the dict to JSON so we can store it as an attribute
        center_points_json = json.dumps(center_points)
        center_points_list.append(center_points_json)

    # Stack the final_labeled_regions along a new time dimension
    final_labeled_regions_array = np.stack(final_labeled_regions_list, axis=0)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {"final_labeled_regions": (["time", "y", "x"], final_labeled_regions_array)},
        coords={
            "time": times,
            "y": np.arange(final_labeled_regions_array.shape[1]),
            "x": np.arange(final_labeled_regions_array.shape[2]),
        },
        attrs={
            "description": "Detection results of MCSs",
            "note": "This file contains the final labeled regions from MCS detection.",
        },
    )

    # Add lat/lon as DataArray variables (assuming lat/lon shape = (y, x))
    ds["lat"] = (("y", "x"), lat)
    ds["lon"] = (("y", "x"), lon)

    # Store the center_points JSON for each timestep as an attribute
    # e.g. "center_points_t0", "center_points_t1", etc.
    for i, cp_json in enumerate(center_points_list):
        ds["final_labeled_regions"].attrs[f"center_points_t{i}"] = cp_json

    # Save to NetCDF file
    ds.to_netcdf(output_filepath)
    print(f"Detection results saved to {output_filepath}")


def load_detection_results(input_filepath):
    """
    Load detection results from a NetCDF file, including each timestep's center-of-mass
    information if present.

    This function looks for JSON attributes named "center_points_t{i}" in the
    "final_labeled_regions" variable for each timestep i, and parses them into a
    dictionary stored in detection_result["center_points"].

    Args:
        input_filepath (str): Path to the input NetCDF file.

    Returns:
        List[dict]: A list of detection_result dictionaries, where each dictionary
            contains:
              - "final_labeled_regions": 2D array of labels (int)
              - "time": Timestamp or datetime-like
              - "lat": 2D array of latitudes
              - "lon": 2D array of longitudes
              - "center_points": (optional) dict { label_value : (center_lat, center_lon) }
                if present in the file.
        or None if loading fails.
    """
    if not os.path.exists(input_filepath):
        print(f"File {input_filepath} does not exist.")
        return None

    try:
        ds = xr.open_dataset(input_filepath)
    except Exception as e:
        print(f"Error opening {input_filepath}: {e}")
        return None

    # Check if required data variables are present
    required_vars = ["final_labeled_regions", "lat", "lon", "time"]
    for var in required_vars:
        if var not in ds.variables and var not in ds.coords:
            print(f"Variable {var} not found in {input_filepath}.")
            return None

    # Extract data
    final_labeled_regions_array = ds["final_labeled_regions"].values
    times = ds["time"].values
    lat = ds["lat"].values
    lon = ds["lon"].values

    # Prepare detection_results list
    detection_results = []
    n_times = final_labeled_regions_array.shape[0]

    for idx in range(n_times):
        # Create a detection result dictionary for this timestep
        time_val = times[idx]
        labeled_regions_2d = final_labeled_regions_array[idx]

        # Attempt to parse center-of-mass JSON attribute for this timestep
        center_key = f"center_points_t{idx}"
        if center_key in ds["final_labeled_regions"].attrs:
            center_points_json = ds["final_labeled_regions"].attrs[center_key]
            try:
                center_points_dict = json.loads(center_points_json)
            except json.JSONDecodeError:
                center_points_dict = {}
        else:
            center_points_dict = {}

        detection_result = {
            "final_labeled_regions": labeled_regions_2d,
            "time": time_val,
            "lat": lat,
            "lon": lon,
            "center_points": center_points_dict
        }
        detection_results.append(detection_result)

    ds.close()
    print(f"Detection results loaded from {input_filepath}")

    return detection_results



def save_tracking_results_to_netcdf(
    mcs_id_list, main_mcs_id_list, lifetime_list, time_list, lat, lon, output_dir
):
    """
    Save tracking results to a NetCDF file.

    Parameters:
    - mcs_id_list: List of MCS detection arrays (binary), each of shape (y, x), incl merging and splitting clusters.
    - main_mcs_id_list: List of MCS ID arrays, each of shape (y, x).
    - time_list: List of timestamps.
    - lat: 2D array of latitudes, shape (y, x).
    - lon: 2D array of longitudes, shape (y, x).
    - output_dir: Directory to save the NetCDF file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Stack the mcs_detected_list and mcs_id_list along the time dimension
    mcs_id = np.stack(mcs_id_list, axis=0)  # Shape: (time, y, x)
    main_mcs_id = np.stack(main_mcs_id_list, axis=0)  # Shape: (time, y, x)
    lifetime_all = np.stack(lifetime_list, axis=0)  # Shape: (time, y, x)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "mcs_id": (["time", "y", "x"], mcs_id),
            "main_mcs_id": (["time", "y", "x"], main_mcs_id),
            "lifetime": (["time", "y", "x"], lifetime_all),
        },
        coords={
            "time": time_list,
            "y": np.arange(lat.shape[0]),
            "x": np.arange(lat.shape[1]),
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        },
    )

    # Set attributes
    ds["mcs_id"].attrs[
        "description"
    ] = "Binary mask of detected MCSs incl merging and splitting clusters"
    ds["main_mcs_id"].attrs[
        "description"
    ] = "Unique IDs of tracked MCSs, only contains main tracks"
    ds["lifetime"].attrs["description"] = "Lifetime of all clusters in time steps"
    ds["lat"].attrs["description"] = "Latitude coordinate"
    ds["lon"].attrs["description"] = "Longitude coordinate"
    ds.attrs["title"] = "MCS Tracking Results"
    ds.attrs[
        "institution"
    ] = "Wegener Center for Global and Climate Change / University of Graz"
    ds.attrs["source"] = "MCS Detection and Tracking Algorithm"
    ds.attrs["history"] = f"Created on {datetime.datetime.now()}"
    ds.attrs["references"] = "David Kneidinger <david.kneidinger@uni-graz.at>"

    # Save to NetCDF file
    output_filepath = os.path.join(output_dir, "mcs_tracking_results.nc")
    ds.to_netcdf(output_filepath)

    print(f"Saved tracking results to {output_filepath}")
