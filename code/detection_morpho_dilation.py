import numpy as np
import xarray as xr
import hdbscan
from scipy.signal import fftconvolve
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    generate_binary_structure,
)
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops, label
from collections import defaultdict
from input_output import load_data


def smooth_precipitation_field(precipitation, kernel_size=2):
    """
    Apply simple box filter using FFT convolution with a kernel_size x kernel_size box to the precipitation field.

    Parameters:
    - precipitation: 2D array of precipitation values.
    - kernel_size: Size of the box filter kernel.

    Returns:
    - Smoothed precipitation field as a 2D array.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    kernel /= kernel.sum()
    return fftconvolve(precipitation, kernel, mode="same")


def detect_cores_hdbscan(precipitation, lat, lon, core_thresh=10.0, min_cluster_size=3):
    """
    Cluster heavy precipitation cores using HDBSCAN.

    Parameters:
    - precipitation: 2D precipitation field.
    - lat: 2D array of latitude values corresponding to the precipitation grid.
    - lon: 2D array of longitude values corresponding to the precipitation grid.
    - core_thresh: Threshold for heavy precipitation cores.
    - min_cluster_size: Minimum number of samples in a cluster for HDBSCAN.

    Returns:
    - labeled_array: 2D array with cluster labels for each grid point.
      Points not belonging to any cluster are labeled as -1.
    """
    """
    Example, same as above but we pass lat2d, lon2d as arguments.
    """
    core_mask = precipitation >= core_thresh
    labels_2d = np.zeros_like(precipitation, dtype=int)
    if np.sum(core_mask) < min_cluster_size:
        return labels_2d

    # Extract lat/lon for the masked pixels
    core_coords = np.column_stack((lat[core_mask].ravel(), lon[core_mask].ravel()))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="haversine", allow_single_cluster=True
    )
    clusterer.fit(np.radians(core_coords))
    # clusterer.labels_ = [-1, 0, 1, 2, ...]
    core_labels = np.where(clusterer.labels_ == 0, -1, clusterer.labels_ + 1)

    # Insert into 2D array
    labels_2d[core_mask] = core_labels
    return labels_2d


def morphological_expansion_with_merging(
    core_labels,
    precip,
    expand_threshold=0.1,
    max_iterations=80
):
    """
    Performs iterative morphological expansion of labeled cores. In each iteration:
      1) Dilate each label by one pixel (8-connected).
      2) Collect newly added pixels (where precip >= expand_threshold).
      3) Build collisions (pixels claimed by multiple labels).
      4) Repeatedly merge collisions until no further merges occur.
      5) Finally apply expansions to the core_labels.
    Repeats until no more pixels are added or until max_iterations is reached.

    Args:
        core_labels (np.ndarray): 2D integer array of core labels (>0) and background=0
        precip (np.ndarray): 2D precipitation array (same shape as core_labels)
        expand_threshold (float): Minimum precipitation required to add new pixels
        max_iterations (int): Maximum expansion iterations

    Returns:
        np.ndarray: Updated label array after expansions and merges

    Notes:
        - Collisions are unified by default if precip >= expand_threshold at the collision pixel.
        - The merges are done in sets (transitive merges).
        - We re-check collisions repeatedly within each iteration to avoid partial merges
          (which can cause 'checkerboard' leftovers).
    """

    structure = generate_binary_structure(2, 1)  # 8-connected
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        changed_pixels_total = 0

        # 1) Gather expansions for each label
        expansions = {lbl: set() for lbl in np.unique(core_labels) if lbl > 0}

        # Morphologically dilate each label by 1 step
        for lbl in expansions:
            feature_mask = (core_labels == lbl)
            dilated_mask = binary_dilation(feature_mask, structure=structure)
            # Only accept new pixels where precip >= expand_threshold
            new_pixels = dilated_mask & (~feature_mask) & (precip >= expand_threshold)

            if np.any(new_pixels):
                coords = zip(*new_pixels.nonzero())
                for (r, c) in coords:
                    expansions[lbl].add((r, c))

            changed_pixels_total += np.count_nonzero(new_pixels)

        if changed_pixels_total == 0:
            print(f"Expansion converged after {iteration} iterations.")
            break

        # 2) Merge collisions in a loop until stable
        merges_happened = True
        while merges_happened:
            merges_happened = False

            # 2a) Build a mapping from pixel -> list of labels claiming it
            pixel_claims = defaultdict(list)
            for lbl, pixset in expansions.items():
                for (r, c) in pixset:
                    pixel_claims[(r, c)].append(lbl)

            # 2b) Detect collisions
            merges = []
            for (r, c), claim_list in pixel_claims.items():
                if len(claim_list) > 1 and precip[r, c] >= expand_threshold:
                    merges.append(set(claim_list))

            if not merges:
                break  # no collisions => done merging

            merges_to_apply = unify_merge_sets(merges)

            # 2c) Apply merges => pick smallest label as master
            for mg in merges_to_apply:
                if len(mg) < 2:
                    continue  # single label
                master = min(mg)
                old_labels = [x for x in mg if x != master]

                # Reassign expansions
                for old_lbl in old_labels:
                    if old_lbl in expansions:
                        expansions[master].update(expansions[old_lbl])
                        del expansions[old_lbl]
                        merges_happened = True

                # Also rewrite label array so we don't keep partial expansions
                # (this ensures collisions are recognized properly if they happen again)
                for old_lbl in old_labels:
                    core_labels[core_labels == old_lbl] = master

        # 3) Finally, apply expansions to core_labels
        #    Now that merges are stable for this iteration
        for lbl, pixset in expansions.items():
            for (r, c) in pixset:
                core_labels[r, c] = lbl

    return core_labels

def unify_merge_sets(merges):
    """
     Merges overlapping sets of labels transitively.
     For example, if merges = [ {1,2}, {2,3}, {4,5}, {1,3} ],
     the end result is [ {1,2,3}, {4,5} ].

    Args:
        merges (list of set): Each set contains labels that must unify.

     Returns:
         list of set: The final merged sets after transitive unification.

      Args:
          merges (list): list of sets, e.g. [ {1,2}, {2,3}, ... ]
    """
    merged = []
    for mset in merges:
        # compare with existing sets in 'merged'
        found = False
        for i, existing in enumerate(merged):
            if mset & existing:
                merged[i] = existing.union(mset)
                found = True
                break
        if not found:
            merged.append(mset)
    # repeat until stable
    stable = False
    while not stable:
        stable = True
        new_merged = []
        for s in merged:
            merged_any = False
            for i, x in enumerate(new_merged):
                if s & x:
                    new_merged[i] = x.union(s)
                    merged_any = True
                    stable = False
                    break
            if not merged_any:
                new_merged.append(s)
        merged = new_merged
    return merged


def unify_checkerboard_simple(core_labels, precip, threshold=0.1, max_passes=10):  # TODO: should not be necessary if morphological_expansion_with_merging is working correctly
    """
    A simpler “lowest‐label‐wins” approach to fix checkerboard patterns by
    repeatedly scanning for local adjacencies where a pixel can unify to a smaller label.

    For each labeled pixel (r, c):
      - Check its 8 neighbors.
      - If any neighbor has a smaller label M < L, and either cell's precipitation
        is >= threshold, unify label L -> M (lowest label wins).
    This repeats up to max_passes times or until no more unifications occur,
    eliminating checkerboard patches.

    Args:
        core_labels (np.ndarray):
            2D integer array (labels > 0, 0 = background).
        precip (np.ndarray):
            2D float array, same shape as core_labels.
        threshold (float):
            Precipitation threshold to allow merging. If either pixel's precip
            is >= threshold, we unify.
        max_passes (int):
            Limit on how many times we loop over the array to unify labels.

    Returns:
        np.ndarray:
            Updated core_labels array with checkerboard boundaries minimized,
            using a local “lowest label wins” rule.

    Notes:
        - Because we unify L -> M if M < L, large clusters with smaller IDs can
          absorb neighboring clusters with bigger IDs if they share a boundary
          above the threshold.
        - In extreme cases, this can unify more than you want. Use with caution.
        - Generally faster and easier than a full adjacency BFS approach, but can
          require several passes for large domains.
    """
    nrows, ncols = core_labels.shape
    changed = True
    passes = 0

    # Offsets for 8-neighborhood
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

    while changed and passes < max_passes:
        changed = False
        passes += 1

        # We'll scan row by row
        for r in range(nrows):
            for c in range(ncols):
                lbl = core_labels[r, c]
                if lbl <= 0:
                    continue

                val_rc = precip[r, c]
                # Check 8 neighbors
                for dr, dc in neighbors_8:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < nrows and 0 <= cc < ncols:
                        neighbor_lbl = core_labels[rr, cc]
                        if neighbor_lbl > 0 and neighbor_lbl < lbl:
                            # Check precipitation threshold
                            val_neighbor = precip[rr, cc]
                            if val_rc >= threshold or val_neighbor >= threshold:
                                # unify lbl -> neighbor_lbl
                                core_labels[core_labels == lbl] = neighbor_lbl
                                changed = True
                                # We must break out after a unify, because 'lbl' is gone
                                break
                if changed:
                    # Once we've changed one pixel in this row, we might as well
                    # proceed to next pixel. The entire cluster lbl might already
                    # be overwritten
                    # So we do break out of the for c loop:
                    break
            if changed:
                # break out of the row loop as well
                break

    return core_labels


def filter_mcs_candidates(
    clusters, convective_plumes, min_area_km2, min_nr_plumes, grid_cell_area_km2
):
    """
    Filter clusters to identify MCS candidates based on area and number of convective plumes.

    Parameters:
    - clusters: 2D array of cluster labels.
    - convective_plumes: 2D array of convective plume labels.
    - min_area_km2: Minimum area threshold for MCS candidate (in km²).
    - min_nr_plumes: Minimum number of convective plumes required for MCS candidate.
    - grid_cell_area_km2: Area of a single grid cell (in km²).

    Returns:
    - mcs_candidate_labels: List of cluster labels that meet the MCS criteria.
    """
    mcs_candidate_labels = []
    cluster_labels = np.unique(clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]

    for label_value in cluster_labels:
        cluster_mask = clusters == label_value
        area_km2 = np.sum(cluster_mask) * grid_cell_area_km2
        plumes_in_cluster = np.unique(convective_plumes[cluster_mask])
        num_plumes = len(plumes_in_cluster[plumes_in_cluster != 0])

        if area_km2 >= min_area_km2 and num_plumes >= min_nr_plumes:
            mcs_candidate_labels.append(label_value)

    return mcs_candidate_labels


def extract_shape_features(clusters, lat, lon, grid_spacing_km):
    """
    Extract shape features from detected clusters.

    Parameters:
    - clusters: 2D array of cluster labels.
    - lat, lon: 2D arrays of latitude and longitude.
    - grid_spacing_km: Approximate grid spacing in kilometers.

    Returns:
    - shape_features: Dictionary with cluster labels as keys and feature dictionaries as values.
    """
    shape_features = {}
    # Label clusters for regionprops
    labeled_clusters = clusters.astype(int)
    cluster_labels = np.unique(labeled_clusters)
    cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background

    for label_value in cluster_labels:
        cluster_mask = labeled_clusters == label_value
        # Convert mask to binary image
        binary_image = cluster_mask.astype(int)

        # Compute region properties
        props_list = regionprops(binary_image)
        if len(props_list) == 0:
            continue  # Skip if no properties are found
        props = props_list[0]  # There should be only one region in the mask

        # Extract features
        area = props.area * (grid_spacing_km**2)  # Convert to km²
        perimeter = props.perimeter * grid_spacing_km  # Convert to km
        major_axis_length = props.major_axis_length * grid_spacing_km
        minor_axis_length = props.minor_axis_length * grid_spacing_km
        aspect_ratio = (
            major_axis_length / minor_axis_length if minor_axis_length != 0 else np.nan
        )
        orientation = props.orientation  # In radians
        solidity = props.solidity  # Convexity
        eccentricity = props.eccentricity  # Elongation measure
        extent = props.extent  # Ratio of area to bounding box area
        convex_area = props.convex_area * (grid_spacing_km**2)
        circularity = (
            (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else np.nan
        )

        # Convert orientation to degrees and adjust range
        orientation_deg = np.degrees(orientation)
        orientation_deg = (orientation_deg + 360) % 360

        shape_features[label_value] = {
            "area_km2": area,
            "perimeter_km": perimeter,
            "major_axis_length_km": major_axis_length,
            "minor_axis_length_km": minor_axis_length,
            "aspect_ratio": aspect_ratio,
            "orientation_deg": orientation_deg,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "extent": extent,
            "convex_area_km2": convex_area,
            "circularity": circularity,
        }

    return shape_features


def classify_mcs_types(shape_features):
    """
    Classify MCS clusters into types based on shape features.

    Parameters:
    - shape_features: Dictionary of shape features per cluster.

    Returns:
    - mcs_classification: Dictionary with cluster labels as keys and MCS types as values.
    """
    mcs_classification = {}

    for label_value, features in shape_features.items():
        aspect_ratio = features["aspect_ratio"]
        area = features["area_km2"]
        circularity = features["circularity"]

        # Initialize type
        mcs_type = "Unclassified"

        # Classification rules
        if aspect_ratio >= 5 and features["major_axis_length_km"] >= 100:
            mcs_type = "Squall Line"
        elif aspect_ratio <= 2 and area >= 100000 and circularity >= 0.7:
            mcs_type = "MCC"
        elif 2 <= aspect_ratio < 5:
            mcs_type = "Linear MCS"
        else:
            mcs_type = "Other MCS"

        mcs_classification[label_value] = mcs_type

    return mcs_classification


def detect_mcs_in_file(
    file_path,
    data_var,
    heavy_precip_threshold,
    moderate_precip_threshold,
    min_size_threshold,
    min_nr_plumes,
    grid_spacing_km,
    time_index=0,
):
    """
    Detect MCSs in a single file using HDBSCAN for clustering.

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
    # Load data
    ds, lat, lon, precipitation = load_data(file_path, data_var, time_index)

    # Step 1: Smooth the precipitation field
    precipitation_smooth = smooth_precipitation_field(precipitation, kernel_size=2)

    # Step 2: Detect heavy precipitation cores with HDBSCAN
    core_labels = detect_cores_hdbscan(
        precipitation,
        lat,
        lon,
        core_thresh=heavy_precip_threshold,
        min_cluster_size=3,  # Min number of points in a cluster
    )

    # Step 3: Morphological expansion with merging
    expanded_labels = morphological_expansion_with_merging(
        core_labels,
        precipitation_smooth,
        expand_threshold=moderate_precip_threshold,
        max_iterations=80,
    )

    expanded_labels = unify_checkerboard_simple(
        expanded_labels,
        precipitation_smooth,
        threshold=moderate_precip_threshold,
        max_passes=10,
    )

    # Step 4: Filter MCS candidates based on number of convective plumes and area
    grid_cell_area_km2 = grid_spacing_km**2
    mcs_candidate_labels = filter_mcs_candidates(
        expanded_labels,
        core_labels,
        min_size_threshold,
        min_nr_plumes,
        grid_cell_area_km2,
    )

    # Create final labeled regions for MCS candidates
    final_labeled_regions = np.where(
        np.isin(expanded_labels, mcs_candidate_labels), expanded_labels, 0
    )

    # Step 7: Extract shape features from clusters
    shape_features = extract_shape_features(
        final_labeled_regions, lat, lon, grid_spacing_km
    )

    # Step 8: Classify MCS types
    mcs_classification = classify_mcs_types(shape_features)

    # Make final labeled regions that are no cluster to be -1
    final_labeled_regions[
        final_labeled_regions == 0
    ] = (
        -1
    )  # TODO: find a better solution, think of using 0 to to be able to save detection results as int

    # Prepare detection result
    detection_result = {
        "file_path": file_path,
        "final_labeled_regions": final_labeled_regions,
        "lat": lat,
        "lon": lon,
        "precipitation": precipitation_smooth,
        "time": ds["time"].values,
        "convective_plumes": core_labels,
        "shape_features": shape_features,
        "mcs_classification": mcs_classification,
    }
    return detection_result
