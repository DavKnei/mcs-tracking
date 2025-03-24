import numpy as np
import xarray as xr
import os
from datetime import datetime, timedelta

def linear_interpolate(t, t_start, t_end, start_val, end_val):
    if t <= t_start:
        return start_val
    if t >= t_end:
        return end_val
    frac = (t - t_start) / (t_end - t_start)
    return start_val + frac * (end_val - start_val)

def create_circular_cell(pr_array, center_y, center_x, radius, heavy_prec, moderate_prec):
    """
    Draw a circular precipitation cell with a gradient from heavy_prec at the center to moderate_prec at the edge.
    """
    ny, nx = pr_array.shape
    y_indices, x_indices = np.indices((ny, nx))
    dist = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
    inside_mask = dist <= radius
    if np.any(inside_mask):
        pr_array[inside_mask] = moderate_prec + (heavy_prec - moderate_prec) * (1 - dist[inside_mask] / radius)

def create_circular_li_field(li_array, center_y, center_x, radius, min_val, max_val):
    """
    Create a circular LI field with a linear gradient from min_val at the center to max_val at the edge.
    """
    ny, nx = li_array.shape
    y_indices, x_indices = np.indices((ny, nx))
    dist = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
    inside_mask = dist <= radius
    if np.any(inside_mask):
        li_array[inside_mask] = min_val + (max_val - min_val) * (dist[inside_mask] / radius)

def create_horizontal_li_bands(li_array, n_bands, band_values):
    """
    Fill li_array with horizontal bands that span the entire domain.
    The vertical dimension is split into n_bands equal parts, and each band is assigned a constant LI value from band_values.
    """
    ny, nx = li_array.shape
    for band in range(n_bands):
        y_start = int(band * ny / n_bands)
        y_end = int((band + 1) * ny / n_bands)
        li_array[y_start:y_end, :] = band_values[band]

def save_single_timestep(out_dir, scenario_name, current_time, lat2d, lon2d, pr_slice, li_slice):
    """
    Save a single timestep of test data as a netCDF file.
    """
    ds = xr.Dataset(
        {
            "pr": (("time", "y", "x"), pr_slice[np.newaxis, ...]),
            "li": (("time", "y", "x"), li_slice[np.newaxis, ...])
        },
        coords={
            "time": [str(current_time)],
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d)
        },
        attrs={
            "description": "Complex scenario with growth, translation, merging, and splitting",
            "author": "David Kneidinger",
            "email": "david.kneidinger@uni-graz.at"
        }
    )
    ds["pr"].attrs["units"] = "mm/h"
    ds["pr"].attrs["long_name"] = "precipitation"
    ds["li"].attrs["units"] = "Kelvin"
    ds["li"].attrs["long_name"] = "lifting index"
    file_name = f"{scenario_name}_{current_time:%Y-%m-%d-%H-00}.nc_test"
    file_path = os.path.join(out_dir, file_name)
    ds.to_netcdf(file_path)

def create_test_data_scenario(out_dir):
    """
    Creates test data:
      - Three circular precipitation systems that have merging and splitting scenarios.
      - One additional large MCS-like precipitation system that mvoes before the three smaller systems. 
      - One extra big system above the three systems. This extra system moves horiontally in space, but has no convective environment.
      - The LI field is defined over the entire domain as horizontal bands with values [-15, -10, -5, -2, 2].
    """
    os.makedirs(out_dir, exist_ok=True)

    # Grid (100 x 100)
    lat = np.arange(-5, 5, 0.1)
    lon = np.arange(0, 10, 0.1)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
    ny, nx = lat2d.shape

    # Time steps: 16 hourly steps
    start_time = datetime(2020, 1, 1, 0, 0)
    nt = 16

    # Precipitation parameters
    heavy_prec = 15.0
    moderate_prec = 5.0

    # Initial positions for three clusters at t=0
    top_init_y = 30
    mid_init_y = 50
    bot_init_y = 70
    init_x = 20

    # Radius evolution for the three clusters
    def radius_func(t):
        if t <= 3:
            return linear_interpolate(t, 0, 3, 2, 7)
        elif t <= 13:
            return 7
        else:
            return linear_interpolate(t, 13, 15, 7, 2)

    # X movement for the three clusters (cells move east)
    def x_pos(t):
        return init_x + 5*t

    # Vertical movement functions for the three clusters
    def bot_y_func(t):
        if t <= 5:
            return 70
        elif t == 6:
            return 60
        elif t == 7:
            return 55  # merged with mid
        elif t <= 10:
            return 55  # stable merged
        elif t == 11:
            return 55
        elif t == 12:
            return 65  # start splitting bottom
        elif t == 13:
            return 70
        else:
            return 70

    def top_y_func(t):
        if t <= 7:
            return 30
        elif t == 8:
            return 45  # merges with mid
        elif t <= 10:
            return 45  # stable merged
        elif t == 11:
            return 30  # splits off
        elif t <= 12:
            return 30
        elif t <= 13:
            return 25  # fully separated
        else:
            return 25

    def mid_y_func(t):
        return 50

    # Extra precipitation system (the one that is non-convective)
    # This system moves in space as the others. We use a similar translation.
    def extra_x_func(t):
        return x_pos(t) + 30
    # For extra_y, let it follow a similar pattern to mid_y.
    def extra_y_func(t):
        return 50  # constant mid-level for extra system

    extra_radius = 8  # smaller than before
    # For the extra system, set LI to a constant value (non-convective): -1.
    
    scenario_name = "MCS-test"

    for t in range(nt):
        current_time = start_time + timedelta(hours=t)
        pr_slice = np.zeros((ny, nx), dtype=np.float32)
        li_slice = np.zeros((ny, nx), dtype=np.float32)

        # --- Create the three circular clusters ---
        r = radius_func(t)
        tx = int(x_pos(t))
        ty_top = top_y_func(t)
        ty_mid = mid_y_func(t)
        ty_bot = bot_y_func(t)

        # Top cluster
        create_circular_cell(pr_slice, ty_top, tx, r, heavy_prec, moderate_prec)
        create_circular_li_field(li_slice, ty_top, tx, r, -15, -2)

        # Middle cluster
        create_circular_cell(pr_slice, ty_mid, tx, r+2, heavy_prec, moderate_prec)
        create_circular_li_field(li_slice, ty_mid, tx, r+2, -15, -2)

        # Bottom cluster
        create_circular_cell(pr_slice, ty_bot, tx, r, heavy_prec, moderate_prec)
        create_circular_li_field(li_slice, ty_bot, tx, r, -15, -2)

        # --- Create an additional MCS-like precipitation system (non-convective) ---
        create_circular_cell(pr_slice, 50, tx+30, r+5, heavy_prec, moderate_prec)
        ny_indices, nx_indices = pr_slice.shape
        y_idx, x_idx = np.indices((ny_indices, nx_indices))
        dist_extra = np.sqrt((y_idx - 50)**2 + (x_idx - (tx+30))**2)
        extra_mask1 = dist_extra <= (r+5)
        li_slice[extra_mask1] = -1  # Not meeting LI < -2

        # --- Create the extra big system above the three clusters ---
        # This extra system moves in space similarly to the others.
        extra_center_x = extra_x_func(t)
        extra_center_y = extra_y_func(t)
        create_circular_cell(pr_slice, 90, tx, 10, heavy_prec, moderate_prec)
        # For the LI field, instead of a circular gradient, create horizontal bands over the entire domain.
        li_extra = np.empty((ny, nx), dtype=np.float32)
        n_bands = 5
        li_band_values = [-15, -10, -5, -2, 2]
        create_horizontal_li_bands(li_extra, n_bands, li_band_values)
        # Override the LI field for the extra system over the whole domain.
        li_slice = li_extra

        save_single_timestep(out_dir, scenario_name, current_time, lat2d, lon2d, pr_slice, li_slice)

    print("Updated test data created successfully at", out_dir)


if __name__ == "__main__":
    create_test_data_scenario("./Test/data/")
