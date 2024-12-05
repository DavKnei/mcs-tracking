import xarray as xr
import os

def split_nc_by_time(input_file, output_dir, variable_name="PR", new_variable_name="pr"):
    """
    Split a NetCDF file by time and save each time step as a separate file, ensuring the time dimension remains.

    Parameters:
    - input_file: str, path to the input .nc file.
    - output_dir: str, directory where output files will be saved.
    - variable_name: str, the name of the variable to rename (default: "PR").
    - new_variable_name: str, the new name for the variable (default: "pr").
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the input NetCDF file
    ds = xr.open_dataset(input_file)
    
    # Ensure the target variable exists
    if variable_name not in ds.variables:
        raise ValueError(f"Variable '{variable_name}' not found in the dataset.")
    
    # Select the target variable and rename it
    ds = ds[[variable_name]].rename({variable_name: new_variable_name})
    
    # Iterate over each time step
    for time in ds.time.values:
        # Select the single time step
        time_step_ds = ds.sel(time=time)
        
        # Ensure the time dimension is retained
        if "time" not in time_step_ds.dims:
            time_step_ds = time_step_ds.expand_dims(dim="time")
        
        # Format the output file name with full date and time
        timestamp = str(time)[:16].replace(":", "-").replace("T", "-")  # e.g., 20200101-10-00
        output_file = os.path.join(output_dir, f"MCS-test_{timestamp}.nc")
        
        # Save to a new NetCDF file
        time_step_ds.to_netcdf(output_file)
        print(f"Saved: {output_file}")

# Example usage
input_file = "MCS-test-4_20200101.nc"
output_dir = "./Test4"
split_nc_by_time(input_file, output_dir)
