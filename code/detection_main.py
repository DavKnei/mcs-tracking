import numpy as np
from skimage.measure import label as connected_label
import logging
from input_output import load_data
from detection_helper_func import (
    smooth_precipitation_field,
    detect_cores_connected,
    morphological_expansion_with_merging,
    unify_checkerboard_simple,
    cascading_threshold_expansion
)
from detection_filter_func import (
    filter_mcs_candidates,
    compute_cluster_centers_of_mass,
)


def detect_mcs_in_file(
    file_path,
    data_var,
    lat_name,
    lon_name,
    heavy_precip_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    grid_spacing_km,
    time_index=0,
):
    """
    Detect MCSs in a single file.

    Parameters:
    - file_path: Path to the NetCDF file containing precipitation data.
    - data_var: Variable name of detected variable.
    - heavy_precip_threshold: Threshold for heavy precipitation (mm/h).
    - moderate_precip_threshold: Threshold for moderate precipitation (mm/h).
    - min_size_threshold: Minimum size threshold for clusters (number of grid cells).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - grid_spacing_km: Approximate grid spacing in kilometers.
    - time_index: Index of the time step to process.

    Returns:
    - detection_result: Dictionary containing detection results.
    """
    logger = logging.getLogger(__name__)

    # Load data
    ds, lat, lon, precipitation = load_data(
        file_path, data_var, lat_name, lon_name, time_index
    )

    # Step 1: Smooth the precipitation field
    precipitation_smooth = smooth_precipitation_field(precipitation, kernel_size=2)

    # First, create a base mask for precipitation >= 1 mm/h.
    
    base_mask = precipitation_smooth >= 1.0
    # Label contiguous regions from the base mask.
    region_labels = connected_label(base_mask, connectivity=2)
    
    # Initialize an empty array for refined labels.
    cascading_labels = np.zeros_like(precipitation_smooth, dtype=int)
    unique_regions = np.unique(region_labels)
    current_max_label = 0
    # Process each contiguous region individually.
    for region in unique_regions:
        if region == 0:
            continue
        region_mask = region_labels == region
        # Apply cascading threshold expansion on this region.
        refined_region = cascading_threshold_expansion(region_mask, precipitation_smooth, low_pct=0.11, high_pct=0.33, base_thresh=1.0, heavy_precip_threshold=8, max_iterations=400)
        # Offset labels to ensure global uniqueness.
        if np.max(refined_region) > 0:
            refined_region[refined_region > 0] += current_max_label
            current_max_label = np.max(refined_region)
        # Merge the refined region into the global label array.
        cascading_labels[region_mask] = refined_region[region_mask]
    # Use the cascading_labels as the detected convective core labels.
    final_labels = unify_checkerboard_simple(
        cascading_labels,
        precipitation_smooth,
        threshold=moderate_precip_threshold,
        max_passes=10,
    )

    # Step 4: Filter MCS candidates based on number of convective plumes and area
    grid_cell_area_km2 = grid_spacing_km**2
    mcs_candidate_labels = filter_mcs_candidates(
        final_labels,
        #core_labels,
        min_size_threshold,
        min_nr_plumes,
        grid_cell_area_km2,
    )

    # Create final labeled regions for MCS candidates
    final_labeled_regions = np.where(
        np.isin(final_labels, mcs_candidate_labels), final_labels, 0
    )

    # Step 5: Compute cluster centers of mass
    cluster_centers = compute_cluster_centers_of_mass(
        final_labeled_regions, lat, lon, precipitation
    )

    # Prepare detection result
    detection_result = {
        "file_path": file_path,
        "final_labeled_regions": final_labeled_regions,
        "lat": lat,
        "lon": lon,
        "precipitation": precipitation_smooth,
        "time": ds["time"].values,
        "center_points": cluster_centers,
    }

    logger.info(f"MCS detection completed for {file_path}.")
    return detection_result
