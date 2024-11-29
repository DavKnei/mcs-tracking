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
    lat = ds['lat']
    lon = ds['lon']
    prec = ds['pr']
    return ds, lat, lon, prec
