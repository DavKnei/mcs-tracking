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
    return start_val + frac*(end_val - start_val)

def create_circular_cell(pr_array, center_y, center_x, radius, heavy_prec, moderate_prec):
    ny, nx = pr_array.shape
    y_indices, x_indices = np.indices((ny, nx))
    dist = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
    inside_mask = dist <= radius
    if np.any(inside_mask):
        pr_array[inside_mask] = moderate_prec + (heavy_prec - moderate_prec)*(1 - dist[inside_mask]/radius)

def save_single_timestep(out_dir, scenario_name, current_time, lat2d, lon2d, pr_slice):
    ds = xr.Dataset(
        {
            "pr": (("time","y","x"), pr_slice[np.newaxis, ...])
        },
        coords={
            "time": [str(current_time)],
            "lat": (("y","x"), lat2d),
            "lon": (("y","x"), lon2d)
        },
        attrs={
            "description": "complex scenario with growth, translation, merging, and splitting",
            "author": "David Kneidinger",
            "email": "david.kneidinger@uni-graz.at"
        }
    )
    ds["pr"].attrs["units"] = "mm/h"
    ds["pr"].attrs["long_name"] = "precipitation"

    file_name = f"{scenario_name}_{current_time:%Y-%m-%d-%H-00}.nc_test"
    file_path = os.path.join(out_dir, file_name)
    ds.to_netcdf(file_path)

def create_test_data_scenario7(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Grid
    lat = np.arange(-5,5,0.1)  # 100 points in lat
    lon = np.arange(0,10,0.1)  # 100 points in lon
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')
    ny, nx = lat2d.shape

    # Time steps: t=0 to t=15 (16 steps)
    start_time = datetime(2020,1,1,0,0)
    nt = 16

    # Precip parameters
    heavy_prec = 15.0
    moderate_prec = 5.0

    # Initial positions:
    # top cluster: y=30, middle: y=50, bottom: y=70 at t=0
    # x start at 20
    top_init_y = 30
    mid_init_y = 50
    bot_init_y = 70
    init_x = 20

    # Radius evolution:
    # t=0 to t=3: radius 2→7
    # t=3 to t=10: radius stable at 7
    # t=10 to t=13: still 7 (merged period)
    # t=13 to t=15: radius 7→2 shrink
    def radius_func(t):
        if t <= 3:
            return linear_interpolate(t,0,3,2,7)
        elif t <= 13:
            return 7
        else:
            return linear_interpolate(t,13,15,7,2)

    # X movement: 5 cells/hour east
    def x_pos(t):
        return init_x + 5*t

    # Vertical movements (merging and splitting):
    # Timeline:
    # t=0 to t=3: grow radius, no vertical move
    # t=4, t=5: stable and translating
    # Start merging at t=6:
    # Bottom merges first at t=7, then top merges at t=8
    # After merged at t=9, t=10 stable merged
    # Splitting starts after t=10:
    # t=11 top splits away first
    # t=12 bottom splits away next
    # t=13 to t=15 shrink radius

    # Bottom cluster merging:
    # Initially bot_y=70 (t<=5 no change)
    # At t=6 start moving bottom upward, by t=7 close enough to mid (y=50)
    # Let's move bottom from y=70 at t=5 to y=55 at t=7
    # t=6: y=60
    # t=7: y=55 merged with mid_y=50 (within radius 7)
    # After merging at t=8 we have top merging, so bottom stays ~55 until t=10 stable
    # After t=10 splitting top first at t=11, no need to move bottom yet
    # At t=12 bottom splits downward again:
    # move bottom from 55 at t=10 to 70 at t=12
    def bot_y_func(t):
        if t<=5:
            return 70
        elif t==6:
            return 60
        elif t==7:
            return 55  # merged with mid
        elif t<=10:
            return 55  # stable merged
        elif t==11:
            return 55  # top splits first, bottom still merged
        elif t==12:
            return 65  # start splitting bottom
        elif t==13:
            return 70 # fully separated
        else:
            return 70

    # Top cluster merging:
    # top_y=30 initially
    # Does not move until bottom merges
    # Bottom merges at t=7, next step top merges at t=8
    # So at t=8 top moves down:
    # at t=7 top=30
    # at t=8 top=45 (close to mid=50)
    # stable merged t=9,t=10 top_y=45
    # Splitting at t=11 top splits first:
    # t=11 top_y=35 (away from mid=50 by 15 cells)
    # t=12 still top_y=35
    # t=13 top_y=30 final separated
    def top_y_func(t):
        if t<=7:
            return 30
        elif t==8:
            return 45  # merges with mid
        elif t<=10:
            return 45  # stable merged
        elif t==11:
            return 30  # top splits first
        elif t<=12:
            return 30
        elif t<=13:
            return 25 # fully separated at t=13
        else:
            return 25

    # Middle cluster stays at y=50 always:
    def mid_y_func(t):
        return 50

    scenario_name = "MCS-test7"

    for t in range(nt):
        current_time = start_time + timedelta(hours=t)
        pr_slice = np.zeros((ny, nx), dtype=np.float32)

        r = radius_func(t)
        tx = int(x_pos(t))

        ty_top = top_y_func(t)
        ty_mid = mid_y_func(t)
        ty_bot = bot_y_func(t)

        # Create three clusters at each time step
        create_circular_cell(pr_slice, ty_top, tx, r, heavy_prec, moderate_prec)
        create_circular_cell(pr_slice, ty_mid, tx, r+2, heavy_prec, moderate_prec)
        create_circular_cell(pr_slice, ty_bot, tx, r, heavy_prec, moderate_prec)
        # Create and additional MCS cell 5 cells to the east
        create_circular_cell(pr_slice, 50, tx+30, r + 5, heavy_prec, moderate_prec)
        
        save_single_timestep(out_dir, scenario_name, current_time, lat2d, lon2d, pr_slice)

    print("Updated test data for scenario 7 created successfully at", out_dir)


create_test_data_scenario7("./Test7/data/")