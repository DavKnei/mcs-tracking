import xarray as xr
import numpy as np
import pandas as pd
import os
import datetime
import json


def convert_precip_units(prec, target_unit="mm/h"):
    """
    Convert the precipitation DataArray to the target unit.

    Recognized unit conversions:
      - 'm', 'meter', 'metre': multiply by 1000 (assumed hourly accumulation)
      - 'kg m-2 s-1': multiply by 3600 (from mm/s to mm/h, given 1 kg/m² = 1 mm water)
      - 'mm', 'mm/h', 'mm/hr': no conversion needed

    Parameters:
    - prec: xarray DataArray of precipitation values.
    - target_unit: Desired unit for the output (default: "mm/h").

    Returns:
    - new_prec: DataArray with converted values and updated units attribute.
    """
    orig_units = prec.attrs.get("units", "").lower()

    if orig_units in ["m", "meter", "metre"]:
        factor = 1000.0
    elif orig_units in ["kg m-2 s-1"]:
        factor = 3600.0
    elif orig_units in ["mm", "mm/h", "mm/hr", "kg m-2"]:
        factor = 1.0
    else:
        print(
            f"Warning: Unrecognized precipitation units '{orig_units}'. No conversion applied."
        )
        factor = 1.0

    new_prec = prec * factor
    new_prec.attrs["units"] = target_unit
    return new_prec


def convert_lifting_index_units(li, target_unit="K"):
    """
    Convert the lifting index DataArray to the target unit.

    Recognized unit conversions:
      - degree Celcius to K

    Parameters:
    - li: xarray DataArray of precipitation values.
    - target_unit: Desired unit for the output (default: "K").

    Returns:
    - new_prec: DataArray with converted values and updated units attribute.
    """
    orig_units = li.attrs.get("units", "")

    if orig_units in ["K", "Kelvin"]:
        constant = 0
    elif orig_units in ["°C", "degree_Celcius"]:
        constant = 273.15
    else:
        print(
            f"Warning: Unrecognized precipitation units '{orig_units}'. No conversion applied."
        )
        factor = 1.0

    new_li = li + constant
    new_li.attrs["units"] = target_unit
    return new_li


def load_precipitation_data(file_path, data_var, lat_name, lon_name, time_index=0):
    """
    Load the dataset and select the specified time step, scaling the precipitation
    variable to units of mm/h for consistency with the detection threshold.

    Parameters:
    - file_path: Path to the NetCDF file.
    - data_var: Name of the precipitation variable.
    - lat_name: Name of the latitude variable.
    - lon_name: Name of the longitude variable.
    - time_index: Index of the time step to select.

    Returns:
    - ds: xarray Dataset for the selected time.
    - lat: 2D array of latitudes.
    - lon: 2D array of longitudes.
    - prec: 2D DataArray of precipitation values (scaled to mm/h).
    """
    ds = xr.open_dataset(file_path)
    ds = ds.isel(time=time_index)  # Select the specified time step
    ds["time"] = ds["time"].values.astype("datetime64[ns]")

    latitude = ds[lat_name].values
    longitude = ds[lon_name].values

    # Ensure lat and lon are 2D arrays.
    if latitude.ndim == 1 and longitude.ndim == 1:
        lon, lat = np.meshgrid(longitude, latitude)
    else:
        lat, lon = latitude, longitude

    prec = ds[str(data_var)]
    return ds, lat, lon, prec


def load_lifting_index_data(file_path, data_var, lat_name, lon_name, time_index=0):
    """
    Load the dataset and select the specified time step, scaling the lifting_index data
    variable to units of K for consistency with the detection threshold.

    Parameters:
    - file_path: Path to the NetCDF file.
    - data_var: Name of the precipitation variable.
    - lat_name: Name of the latitude variable.
    - lon_name: Name of the longitude variable.
    - time_index: Index of the time step to select.

    Returns:
    - ds: xarray Dataset for the selected time.
    - lat: 2D array of latitudes.
    - lon: 2D array of longitudes.
    - prec: 2D DataArray of precipitation values (scaled to mm/h).
    """
    ds = xr.open_dataset(file_path)
    ds = ds.isel(time=time_index)  # Select the specified time step
    ds["time"] = ds["time"].values.astype("datetime64[ns]")

    latitude = ds[lat_name].values
    longitude = ds[lon_name].values

    # Ensure lat and lon are 2D arrays.
    if latitude.ndim == 1 and longitude.ndim == 1:
        lon, lat = np.meshgrid(longitude, latitude)
    else:
        lat, lon = latitude, longitude

    li = ds[str(data_var)]
    # Convert the lifting index data to K using the separate conversion function.
    li = convert_lifting_index_units(li, target_unit="K")

    # Remove all non relevant data variables from dataset
    data_vars_list = [data_var for data_var in ds.data_vars]
    data_vars_list.remove(data_var)
    ds = ds.drop_vars(data_vars_list)

    return ds, lat, lon, li


def serialize_center_points(center_points):
    """Convert a center_points dict with float32 lat/lon to Python floats so json.dumps() works."""
    casted_dict = {}
    for label_str, (lat_val, lon_val) in center_points.items():
        # Convert float32 -> float
        casted_dict[label_str] = (float(lat_val), float(lon_val))
    return json.dumps(casted_dict)


def save_detection_results(detection_results, output_filepath, data_source):
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
        data_source (str):
            Name of the original precipitation data source for attrs of output .nc file (e.g. INCA).
    """
    times = []
    final_labeled_regions_list = []
    lifting_index_regions_list = []
    lat = None
    lon = None
    center_points_list = []  # Will store JSON-encoded center points

    for detection_result in detection_results:
        # Extract the required info
        times.append(
            pd.to_datetime(detection_result["time"]).round("S")
        )  # Round the time values to the nearest second

        final_labeled_regions_list.append(detection_result["final_labeled_regions"])
        lifting_index_regions_list.append(detection_result["lifting_index_regions"])

        if lat is None:
            lat = detection_result["lat"]
            lon = detection_result["lon"]

        # If 'center_points' is present, store it; else store empty
        if "center_points" in detection_result:
            center_points = detection_result["center_points"]
        else:
            center_points = {}

        # Convert the dict to JSON so we can store it as an attribute
        center_points_str = serialize_center_points(center_points)
        center_points_json = json.dumps(center_points_str)
        center_points_list.append(center_points_json)

    # Stack the final_labeled_regions along a new time dimension
    final_labeled_regions_array = np.stack(final_labeled_regions_list, axis=0)
    lifting_index_regions_array = np.stack(lifting_index_regions_list, axis=0)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "final_labeled_regions": (["time", "y", "x"], final_labeled_regions_array),
            "lifting_index_regions": (["time", "y", "x"], lifting_index_regions_array),
        },
        coords={
            "time": times,
            "y": np.arange(final_labeled_regions_array.shape[1]),
            "x": np.arange(final_labeled_regions_array.shape[2]),
        },
        attrs={
            "description": "Detection results of MCSs",
            "source": data_source,
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


def save_single_detection_result(detection_result, output_dir, data_source):
    """Saves a single timestep's detection results to a dedicated NetCDF file.

    The output filename is generated from the result's timestamp.

    Args:
        detection_result (dict):
            A dictionary containing the detection results for one timestep.
            Must contain: "final_labeled_regions", "lifting_index_regions", "lat", "lon", "time",
            and optionally "center_points".
        output_dir (str):
            The directory where the output NetCDF file will be saved.
        data_source (str):
            Name of the original data source for the file's metadata.
    """
    # Extract the timestamp and format it for the filename
    time_val = pd.to_datetime(detection_result["time"]).round("S")
    filename = f"detection_{time_val.strftime('%Y%m%dT%H%M')}.nc"

    output_filepath = os.path.join(output_dir, filename)

    # Prepare data arrays
    final_labeled_regions = np.expand_dims(
        detection_result["final_labeled_regions"], axis=0
    )
    lifting_index_regions = np.expand_dims(
        detection_result["lifting_index_regions"], axis=0
    )
    lat = detection_result["lat"]
    lon = detection_result["lon"]

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "final_labeled_regions": (["time", "y", "x"], final_labeled_regions),
            "lifting_index_regions": (["time", "y", "x"], lifting_index_regions),
        },
        coords={
            "time": [time_val],
            "y": np.arange(final_labeled_regions.shape[1]),
            "x": np.arange(final_labeled_regions.shape[2]),
        },
        attrs={
            "description": "Detection results of MCSs for a single timestep.",
            "source": data_source,
        },
    )

    # Add lat/lon as DataArray variables
    ds["lat"] = (("y", "x"), lat)
    ds["lon"] = (("y", "x"), lon)

    # Handle center points if they exist
    center_points = detection_result.get("center_points", {})
    center_points_str = serialize_center_points(center_points)
    center_points_json = json.dumps(center_points_str)
    ds["final_labeled_regions"].attrs["center_points_t0"] = center_points_json

    # Save to NetCDF file
    ds.to_netcdf(output_filepath)
    # Using print instead of logger here as this might be called from a parallel process
    print(f"Single detection result saved to {output_filepath}")


def load_detection_results(input_filepath, USE_LIFTING_INDEX):
    """
    Load detection results from a NetCDF file, including each timestep's center-of-mass
    information if present.

    This function looks for JSON attributes named "center_points_t{i}" in the
    "final_labeled_regions" variable for each timestep i, and parses them into a
    dictionary stored in detection_result["center_points"].

    Args:
        input_filepath (str): Path to the input NetCDF file.
        USE_LIFTING_INDEX (boolean): Flag if lifting_index_regions is present or not

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

    if USE_LIFTING_INDEX:
        required_vars = [
            "final_labeled_regions",
            "lifting_index_regions",
            "lat",
            "lon",
            "time",
        ]
    else:
        required_vars = ["final_labeled_regions", "lat", "lon", "time"]

    for var in required_vars:
        if var not in ds.variables and var not in ds.coords:
            print(f"Variable {var} not found in {input_filepath}.")
            return None

    if USE_LIFTING_INDEX:
        lifting_index_regions_array = ds["lifting_index_regions"].values

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
                # First decoding attempt
                center_points_intermediate = json.loads(center_points_json)
                # If the result is still a string, decode it again
                if isinstance(center_points_intermediate, str):
                    center_points_dict = json.loads(center_points_intermediate)
                else:
                    center_points_dict = center_points_intermediate
            except json.JSONDecodeError:
                center_points_dict = {}
        else:
            center_points_dict = {}

        if USE_LIFTING_INDEX:
            detection_result = {
                "lifting_index_regions": lifting_index_regions_array[idx],
                "final_labeled_regions": labeled_regions_2d,
                "time": time_val,
                "lat": lat,
                "lon": lon,
                "center_points": center_points_dict,
            }
        else:
            detection_result = {
                "final_labeled_regions": labeled_regions_2d,
                "time": time_val,
                "lat": lat,
                "lon": lon,
                "center_points": center_points_dict,
            }
        detection_results.append(detection_result)

    ds.close()
    print(f"Detection results loaded from {input_filepath}")
    return detection_results


def save_tracking_results_to_netcdf(
    robust_mcs_ids_list,
    mcs_id_list,
    main_mcs_id_list,
    lifetime_list,
    time_list,
    lat,
    lon,
    tracking_centers_list,
    output_dir,
    data_source,
):
    """
    Save tracking results (including center points) to a NetCDF file.

    This function stacks the following data along the time dimension:
      - mcs_id_list: Full MCS track IDs (including merges/splits)
      - main_mcs_id_list: Filtered MCS track IDs for main systems
      - lifetime_list: Per-pixel lifetime arrays

    In addition, it stores the track centers for each timestep in two sets of
    JSON attributes:
      1. center_points_t{i}   => The full center dictionary for all track IDs
      2. center_points_t{i} => A filtered center dictionary only for the main MCS IDs

    Args:
        mcs_id_list (List[numpy.ndarray]):
            List of 2D arrays (y, x) with full MCS track IDs (merges/splits included).
        main_mcs_id_list (List[numpy.ndarray]):
            List of 2D arrays (y, x) with only main MCS track IDs.
        lifetime_list (List[numpy.ndarray]):
            List of 2D arrays (y, x) of per-pixel lifetime values.
        time_list (List[datetime.datetime]):
            List of timestamps (one for each timestep).
        lat (numpy.ndarray):
            2D array of latitudes, shape (y, x).
        lon (numpy.ndarray):
            2D array of longitudes, shape (y, x).
        tracking_centers_list (List[dict]):
            A list (one entry per timestep) of dictionaries mapping:
                track_id -> (center_lat, center_lon).
            This is the full set of track centers for merges/splits.
        output_dir (str):
            Directory in which to save the NetCDF file.
        data_source (str):
            Name of the original precipitation data source for attrs of output .nc file (e.g. INCA).


    Returns:
        None. Writes a file named "mcs_tracking_results.nc" in the output_dir.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Round timelist to seconds
    time_list = pd.to_datetime(time_list).round("S")

    # Stack arrays along the time dimension
    mcs_id = np.stack(mcs_id_list, axis=0)  # (time, y, x)
    main_mcs_id = np.stack(main_mcs_id_list, axis=0)  # (time, y, x)
    robust_mcs_id = np.stack(robust_mcs_ids_list, axis=0)  # (time, y, x)
    lifetime_all = np.stack(lifetime_list, axis=0)  # (time, y, x)

    # Create an xarray Dataset
    ds = xr.Dataset(
        {
            "mcs_id": (["time", "y", "x"], mcs_id),
            "main_mcs_id": (["time", "y", "x"], main_mcs_id),
            "robust_mcs_id": (["time", "y", "x"], robust_mcs_id),
            "lifetime": (["time", "y", "x"], lifetime_all),
        },
        coords={
            "time": time_list,
            "y": np.arange(lat.shape[0]),
            "x": np.arange(lat.shape[1]),
        },
    )

    # Attach lat/lon as data variables
    ds["lat"] = (("y", "x"), lat)
    ds["lon"] = (("y", "x"), lon)

    # Set attributes for each variable
    ds["mcs_id"].attrs[
        "description"
    ] = "Track IDs of all MCSs (including merges and splits)"
    ds["main_mcs_id"].attrs["description"] = "Track IDs of main MCSs (filtered subset)"
    ds["robust_mcs_id"].attrs[
        "description"
    ] = "Track IDs of main MCSs that fulfill lifting index criteria"
    ds["lifetime"].attrs["description"] = "Lifetime of all clusters (in timesteps)"
    ds["lat"].attrs["description"] = "Latitude coordinate"
    ds["lon"].attrs["description"] = "Longitude coordinate"

    # Set global attributes
    ds.attrs["title"] = "MCS Tracking Results"
    ds.attrs[
        "institution"
    ] = "Wegener Center for Global and Climate Change / University of Graz"
    ds.attrs["source"] = data_source
    ds.attrs["history"] = f"Created on {datetime.datetime.now()}"
    ds.attrs["references"] = "David Kneidinger <david.kneidinger@uni-graz.at>"

    # Store center points in JSON attributes for each timestep
    # 1) Full track center points: center_points_t{i}
    # 2) Main track center points: center_points_t{i}, filtered by main_mcs_id
    num_timesteps = len(time_list)
    for i in range(num_timesteps):
        # Get the full center points for timestep i.
        full_centers_dict = tracking_centers_list[
            i
        ]  # e.g. { "101": (lat, lon), "102": (lat, lon), ... }
        full_centers_serialized = serialize_center_points(full_centers_dict)
        full_centers_json = json.dumps(full_centers_serialized)
        ds["mcs_id"].attrs[f"center_points_t{i}"] = full_centers_json

        # For main_mcs_id: filter full centers to those track IDs that appear in main_mcs_id.
        main_ids_2d = ds["main_mcs_id"].isel(time=i).values
        used_main_ids = np.unique(main_ids_2d)
        used_main_ids = used_main_ids[used_main_ids != 0]
        main_centers_dict = {}
        for tid in used_main_ids:
            tid_str = str(tid)
            if tid_str in full_centers_dict:
                main_centers_dict[tid_str] = full_centers_dict[tid_str]
        main_centers_serialized = serialize_center_points(main_centers_dict)
        main_centers_json = json.dumps(main_centers_serialized)
        ds["main_mcs_id"].attrs[f"center_points_t{i}"] = main_centers_json

        # For robust_mcs_id: filter full centers to those track IDs that appear in robust_mcs_id.
        robust_ids_2d = ds["robust_mcs_id"].isel(time=i).values
        used_robust_ids = np.unique(robust_ids_2d)
        used_robust_ids = used_robust_ids[used_robust_ids != 0]
        robust_centers_dict = {}
        for tid in used_robust_ids:
            tid_str = str(tid)
            if tid_str in full_centers_dict:
                robust_centers_dict[tid_str] = full_centers_dict[tid_str]
        robust_centers_serialized = serialize_center_points(robust_centers_dict)
        robust_centers_json = json.dumps(robust_centers_serialized)
        ds["robust_mcs_id"].attrs[f"center_points_t{i}"] = robust_centers_json

    # Save the NetCDF file
    output_filepath = os.path.join(output_dir, "mcs_tracking_results.nc")
    ds.to_netcdf(output_filepath)
    print(f"Saved tracking results to {output_filepath}")
