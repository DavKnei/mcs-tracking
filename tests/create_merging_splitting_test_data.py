import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta

def create_circular_cell(pr_array, center_y, center_x, radius, heavy_prec, moderate_prec):
    """
    Create a circular convective cell with heavy precipitation in the center
    and moderate precipitation at the boundary of the radius.
    pr_array: 2D array to modify
    center_y, center_x: center of the cell in array indices (int)
    radius: radius in grid cells
    heavy_prec: max precipitation at center
    moderate_prec: precipitation at boundary
    """
    ny, nx = pr_array.shape
    y_indices, x_indices = np.indices((ny, nx))
    dist = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
    inside_mask = dist <= radius
    # Linear gradient: dist=0 -> heavy_prec, dist=radius -> moderate_prec
    # Value = moderate_prec + (heavy_prec - moderate_prec)*(1 - dist/radius)
    if np.any(inside_mask):
        pr_array[inside_mask] = moderate_prec + (heavy_prec - moderate_prec)*(1 - dist[inside_mask]/radius)

def save_single_timestep(file_name, time_val, lat2d, lon2d, pr_slice, scenario_desc):
    """
    Save a single time step to a NetCDF file.
    file_name: str - output filename
    time_val: datetime - time of this timestep
    lat2d, lon2d: 2D arrays of lat/lon
    pr_slice: 2D precipitation array for this timestep
    scenario_desc: str - description of scenario (splitting or merging)
    """
    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "pr": (("time", "y", "x"), pr_slice[np.newaxis, ...])
        },
        coords={
            "time": [time_val],
            "lat": (("y","x"), lat2d),
            "lon": (("y","x"), lon2d)
        },
        attrs={
            "description": scenario_desc,
            "author": "Your Name"
        }
    )
    ds["pr"].attrs["units"] = "mm/h"
    ds["pr"].attrs["long_name"] = "precipitation"
    ds.to_netcdf(file_name)

def create_test_data_splitting(out_dir):
    """
    Create test data for splitting scenario (Test5):
    - Duration: 12 hours (t=0 to t=11), hourly steps.
    - Start with one big cell at center.
    - After half the steps (around t=6), it splits into two cells drifting apart.
    - Cells move eastward by 1 cell/hour.
    - At end, two distinct cells well-separated.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Grid definition
    lat = np.arange(-5,5,0.1)
    lon = np.arange(0,10,0.1)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
    ny, nx = lat2d.shape

    # Time setup: 12 hourly timesteps from 2020-01-01 00:00:00
    start_time = datetime(2020,1,1,0,0)
    nt = 12

    # Precip parameters
    heavy_prec = 15.0
    moderate_prec = 5.0
    radius = 10

    center_y, center_x = ny//2, nx//2

    # Movement: each hour shift x by +1 cell (eastward)
    # Splitting: first half (t=0 to t=5): one single cell
    # t=6 to t=11: two cells separated along y
    # Let's shift y positions slightly as we go after splitting:
    # At t=0 single cell at (center_y, center_x)
    # At t=6 split into two cells: one goes upward, one downward
    # By t=11 they are separated by ~20 cells in y
    # Let's linearly interpolate their separation:
    # For simplicity, from t=6 to t=11, top cell moves from center_y to center_y-10
    # bottom cell moves from center_y to center_y+10

    for t in range(nt):
        current_time = start_time + timedelta(hours=t)
        pr_slice = np.zeros((ny, nx), dtype=np.float32)
        # horizontal movement
        current_x = center_x + t  # move east by 1 cell/hour

        if t <= 5:
            # Single cell
            create_circular_cell(pr_slice, center_y, current_x, radius, heavy_prec, moderate_prec)
        else:
            # Two cells
            frac = (t-6)/(11-6) # goes from 0 at t=6 to 1 at t=11
            top_y = int(center_y - 10*frac)
            bottom_y = int(center_y + 10*frac)
            # Make radius a bit smaller for two cells, say radius=8
            create_circular_cell(pr_slice, top_y, current_x, radius-2, heavy_prec, moderate_prec)
            create_circular_cell(pr_slice, bottom_y, current_x, radius-2, heavy_prec, moderate_prec)

        # File naming: MCS-test5_YYYY-MM-DD-HH-00.nc_test
        file_name = f"MCS-test5_{current_time:%Y-%m-%d-%H-00}.nc_test"
        file_path = os.path.join(out_dir, file_name)
        save_single_timestep(file_path, current_time, lat2d, lon2d, pr_slice, "test data for splitting")

def create_test_data_merging(out_dir):
    """
    Create test data for merging scenario (Test6):
    - Duration: 12 hours (t=0 to t=11)
    - Start with two distinct cells far apart.
    - Over time, they approach each other and by t=6 they merge into one cell.
    - Also move eastward by 1 cell/hour.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Grid definition
    lat = np.arange(-5,5,0.1)
    lon = np.arange(0,10,0.1)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
    ny, nx = lat2d.shape

    # Time setup
    start_time = datetime(2020,1,1,0,0)
    nt = 12

    # Precip parameters
    heavy_prec = 15.0
    moderate_prec = 5.0
    radius = 10

    center_y, center_x = ny//2, nx//2

    # Two cells: top and bottom start far apart and merge at t=6
    # Let's say at t=0 top cell at center_y-10, bottom cell at center_y+10
    # By t=6 they meet at center_y (become one cell)
    # Then remain merged
    # Also move east by 1 cell/hour

    for t in range(nt):
        current_time = start_time + timedelta(hours=t)
        pr_slice = np.zeros((ny, nx), dtype=np.float32)
        current_x = center_x + t  # move eastward

        if t <= 5:
            # Two separate cells
            # Linearly move them closer:
            frac = t/5
            # top cell moves from center_y-10 at t=0 to center_y at t=5
            top_y = int((center_y-20) + 10*frac)
            # bottom cell moves from center_y+10 at t=0 to center_y at t=5
            bottom_y = int((center_y+20) - 10*frac)
            create_circular_cell(pr_slice, top_y, current_x, radius, heavy_prec, moderate_prec)
            create_circular_cell(pr_slice, bottom_y, current_x, radius, heavy_prec, moderate_prec)
        else:
            # After t=6, they are merged into one big cell at center_y
            # Possibly increase radius slightly to indicate a merged larger system
            merged_radius = radius + 2
            create_circular_cell(pr_slice, center_y, current_x, merged_radius, heavy_prec, moderate_prec)

        # File naming: MCS-test6_YYYY-MM-DD-HH-00.nc_test
        file_name = f"MCS-test6_{current_time:%Y-%m-%d-%H-00}.nc_test"
        file_path = os.path.join(out_dir, file_name)
        save_single_timestep(file_path, current_time, lat2d, lon2d, pr_slice, "test data for merging")

# Example usage:
# Create a directory for test outputs, e.g. 'test_outputs'
test_outputs_dir = "test_outputs"
os.makedirs(test_outputs_dir, exist_ok=True)

# Create splitting scenario data
create_test_data_splitting(os.path.join(test_outputs_dir, "splitting_scenario"))

# Create merging scenario data
create_test_data_merging(os.path.join(test_outputs_dir, "merging_scenario"))

print("Test data for splitting (Test5) and merging (Test6) created successfully.")
