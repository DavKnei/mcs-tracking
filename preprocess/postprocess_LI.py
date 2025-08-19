import xarray as xr
import numpy as np
import pandas as pd

def postprocess_tracks(ds_track, ds_li, LI_threshold=-2.0, pre_hours=2, robust_duration=6):
    """
    Process tracks using ERA5 LI.
    
    Parameters:
      ds_track      : xarray.Dataset with variable 'main_mcs_id' (dims: time, lat, lon)
      ds_li         : xarray.Dataset with variable 'LI' (dims: time, lat, lon)
      LI_threshold  : threshold (K) for convective environment (default -2 K)
      pre_hours     : number of hours before track start to include (default 2)
      robust_duration: robust track must have at least this many hours (default 6)
    
    Returns:
      ds_track_out  : input dataset with a new variable 'main_mcs_id_robust'
    """
    # Ensure time coordinates are datetime objects
    track_times = pd.to_datetime(ds_track.time.values)
    li_times = pd.to_datetime(ds_li.time.values)
    
    # Create an output array for robust track IDs (initialize to zero)
    robust_ids = np.zeros(ds_track.main_mcs_id.shape, dtype=ds_track.main_mcs_id.dtype)
    
    # Get unique track IDs (exclude 0 and NaN)
    all_ids = np.unique(ds_track.main_mcs_id.values[~np.isnan(ds_track.main_mcs_id.values)])
    unique_tracks = [int(i) for i in all_ids if int(i) != 0]
    
    print(f"Found {len(unique_tracks)} unique tracks.")
    
    # Loop over each unique track id
    for track in unique_tracks:

        if track > 50:
            continue
        # Create a mask for this track from the tracking dataset
        ds_track_mask = ds_track.where(ds_track.main_mcs_id == track, drop=True)
    
        # Get the time indices where the track exists
        time_values = ds_track_mask.time.values

        if len(time_values) == 0:
            continue
        # t0: the first time step when the track appears
        t0 = time_values[0]
        
        # Define the pre-track window: from (t0 - pre_hours) to t0.
        t_pre_start = t0 - pd.Timedelta(hours=pre_hours)
        # Select times in ds_li within this window.
        pre_mask = (li_times >= t_pre_start) & (li_times < t0)
        pre_times = li_times[pre_mask]
        
        # For the pre-track period, use the spatial extent (mask) of the track at t0.
        # Get the mask for t0 from ds_track.
        t0_idx = np.where(track_times == t0)[0][0]
        spatial_mask_t0 = ds_track_mask.isel(time=t0_idx)
        
        # Compute mean LI over pre-track period:
        pre_li_vals = []
        for t in pre_times:
            # Find index in ds_li
            try:
                li_idx = np.where(li_times == t)[0][0]
            except IndexError:
                continue
            # For time t, use the spatial mask from t0
            li_data = ds_li.isel(time=li_idx)
            masked_li = li_data.where(spatial_mask_t0.lat)
            # Compute spatial mean (ignoring NaNs)
            mean_li = float(masked_li.mean().values)
            pre_li_vals.append((t, mean_li))
        
        # Now, for the track period itself:
        track_li_vals = []
        # Loop over all time indices where the track is present.
        for i, t in enumerate(time_values):
            breakpoint()
            spatial_mask = ds_track_mask.isel(time=t.strftime('%Y-%m-%dT%H:%M:%S.000000000'))
            # Find corresponding time in ds_li (assume same time resolution)
            try:
                li_idx = np.where(li_times == t)[0][0]
            except IndexError:
                continue
           
            li_data = ds_li.isel(time=li_idx)
            masked_li = li_data.where(spatial_mask.lat)
            
            mean_li = float(masked_li.mean().values)
            track_li_vals.append((t, mean_li))
        
        # Combine pre-track and track LI values into one time series
        combined_times = [t for t, li in pre_li_vals] + [t for t, li in track_li_vals]
        combined_li = [li for t, li in pre_li_vals] + [li for t, li in track_li_vals]
        combined_series = pd.Series(data=combined_li, index=combined_times)
        
        # Check: if the condition (mean LI < LI_threshold) is never met, mark track non-robust.
        if not (combined_series < LI_threshold).any():
            continue  # leave robust_ids for this track as 0
        
        # Determine convective onset: earliest time when mean LI < threshold
        convective_onset_time = combined_series[combined_series < LI_threshold].index[0]
        
        # Now, determine the robust period as the portion of the track from convective onset onward.
        # Note: If convective onset occurs before t0, we use t0 as the start of the track.
        robust_start_time = t0 if convective_onset_time < t0 else convective_onset_time
        
        # Find the last time of the track (from track_times)
        track_end_time = time_values[-1]
        duration = (track_end_time - robust_start_time).total_seconds() / 3600.0  # in hours
        
        if duration < robust_duration:
            # Not robust: track does not have at least robust_duration hours after convective onset.
            continue
        
        # Otherwise, mark the robust portion:
        # For every time step in ds_track (using the global time coordinate), if the time >= robust_start_time,
        # then for grid points where the track exists, set robust id = track.
        for idx, t in enumerate(track_times):
            if t >= robust_start_time:
                # Find the overall time index in ds_track (using track_times indices matches ds_track.time ordering)
                # Here, we need to locate indices in ds_track.time that equal this time.
                overall_idx = np.where(track_times == t)[0][0]
                # Actually, we need to loop over ds_track.time for the track.
                # Instead, loop over all time indices in ds_track and check if t >= robust_start_time.
        # A simpler approach is to loop over all time indices in ds_track.time:
        for i, t in enumerate(track_times):
            if t >= robust_start_time:
                # For time index corresponding to track time (using the mask we already have)
                # Find the indices (lat, lon) where ds_track.main_mcs_id==track at this time.
                robust_ids[i, :, :] = np.where(ds_track_mask.main_mcs_id.isel(time=i), track, robust_ids[i, :, :])
        print('Finished with track:', track)      
    return robust_ids

def main():
    # Load datasets
    ds_track = xr.open_dataset("./output/mcs_tracking_results.nc")
    ds_li = xr.open_dataset("./data/ERA5/ERA5_201708_LI.nc").drop('expver')
    ds_li = ds_li.LI.interp(time=ds_track.time, longitude=ds_track.lon, latitude=ds_track.lat, method="nearest")

    # Ensure the tracking dataset has time as datetime (if not already)
    ds_track['time'] = pd.to_datetime(ds_track.time.values)
    ds_li['time'] = pd.to_datetime(ds_li.time.values)
    
    # Process each track and build robust track variable.
    robust_array = postprocess_tracks(ds_track, ds_li, LI_threshold=-2.0, pre_hours=2, robust_duration=6)
    # Add new variable to ds_track
    ds_track["main_mcs_id_robust"] = (("time", "y", "x"), robust_array)
    # Save output
    ds_track.to_netcdf("./output/mcs_tracking_results_li.nc")
    print("Saved robust tracking results as mcs_tracking_results_li.nc")

if __name__ == "__main__":
    main()
