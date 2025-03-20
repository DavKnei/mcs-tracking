import numpy as np


def assign_new_id(
    label,
    cluster_mask,
    area,
    next_cluster_id,
    lifetime_dict,
    max_area_dict,
    mcs_id,
    mcs_lifetime,
):
    """
    Assigns a brand-new ID to a cluster with no overlap from the previous timestep.

    Args:
        label (int): Label of the current cluster (from detection).
        cluster_mask (numpy.ndarray): Boolean mask for the current cluster's pixels.
        area (float): Area (km²) of this cluster.
        next_cluster_id (int): Next available integer ID for track assignment.
        lifetime_dict (dict): Tracks how many timesteps each track ID has existed.
        max_area_dict (dict): Tracks the maximum area encountered by each track ID.
        mcs_id (numpy.ndarray): 2D array where we assign the track ID for each pixel.
        mcs_lifetime (numpy.ndarray): 2D array for per-pixel lifetime assignment.

    Returns:
        assigned_id (int): The newly assigned track ID for this cluster.
        next_cluster_id (int): The updated next cluster ID (incremented by 1).
    """
    mcs_id[cluster_mask] = next_cluster_id
    lifetime_dict[next_cluster_id] = 1
    max_area_dict[next_cluster_id] = area
    mcs_lifetime[cluster_mask] = 1
    assigned_id = next_cluster_id
    next_cluster_id += 1
    return assigned_id, next_cluster_id


def get_dominant_cluster(prev_ids, max_area_dict):
    """
    Finds the 'dominant' cluster (largest area) among a list of track IDs.

    Args:
        prev_ids (List[int]): List of old track IDs (integers).
        max_area_dict (dict): Dictionary mapping track ID -> maximum area found so far.

    Returns:
        best_id (int): The track ID with the largest area in `max_area_dict`.
    """
    best_id = None
    best_area = -1
    for pid in prev_ids:
        if max_area_dict.get(pid, 0) > best_area:
            best_area = max_area_dict[pid]
            best_id = pid
    return best_id


def check_overlaps(
    previous_labeled_regions,
    final_labeled_regions,
    previous_cluster_ids,
    overlap_threshold=10,
):
    """
    Checks overlap between old-labeled regions and new-labeled regions.

    We build a mapping new_label -> list of old track IDs that meet or exceed
    `overlap_threshold` percent overlap with the new cluster.

    Args:
        previous_labeled_regions (numpy.ndarray): Labeled regions from the previous timestep.
            0 means no cluster.
        final_labeled_regions (numpy.ndarray): Labeled regions from the current timestep.
            0 means no cluster.
        previous_cluster_ids (dict): Maps old detection labels -> old track IDs.
        overlap_threshold (float): Minimum percentage overlap required for consideration.

    Returns:
        overlap_map (dict): { new_label (int) : [list of old track IDs (int)] }.
    """
    overlap_map = {}

    unique_prev_labels = np.unique(previous_labeled_regions)
    unique_prev_labels = unique_prev_labels[unique_prev_labels != 0]
    unique_curr_labels = np.unique(final_labeled_regions)
    unique_curr_labels = unique_curr_labels[unique_curr_labels != 0]

    for new_label in unique_curr_labels:
        curr_mask = final_labeled_regions == new_label
        curr_area = np.sum(curr_mask)

        relevant_old_ids = []
        if curr_area == 0:
            overlap_map[new_label] = relevant_old_ids
            continue

        for old_label_detection in unique_prev_labels:
            old_track_id = previous_cluster_ids.get(old_label_detection, None)
            if old_track_id is None:
                continue
            prev_mask = previous_labeled_regions == old_label_detection
            overlap_pixels = np.logical_and(curr_mask, prev_mask)
            overlap_area = np.sum(overlap_pixels)
            if overlap_area > 0:
                overlap_percent = (overlap_area / curr_area) * 100
                if overlap_percent >= overlap_threshold:
                    relevant_old_ids.append(old_track_id)
        overlap_map[new_label] = relevant_old_ids

    return overlap_map


def handle_no_overlap(
    new_labels_no_overlap,
    final_labeled_regions,
    next_cluster_id,
    lifetime_dict,
    max_area_dict,
    mcs_id,
    mcs_lifetime,
    grid_cell_area_km2,
):
    """
    Assigns brand-new track IDs to all new labels that found no overlap with previous clusters.

    Args:
        new_labels_no_overlap (List[int]): All new detection labels that found no old ID.
        final_labeled_regions (numpy.ndarray): Labeled regions from the current timestep.
        next_cluster_id (int): Next available integer track ID.
        lifetime_dict (dict): Track ID -> lifetime.
        max_area_dict (dict): Track ID -> max area encountered.
        mcs_id (numpy.ndarray): 2D array for per-pixel track IDs.
        mcs_lifetime (numpy.ndarray): 2D array for per-pixel lifetime values.
        grid_cell_area_km2 (float): Multiplicative factor to get area (km²).

    Returns:
        assigned_ids_map (dict): new_label -> assigned track ID
        next_cluster_id (int): Updated ID counter.
    """
    assigned_ids_map = {}

    for lbl in new_labels_no_overlap:
        mask = final_labeled_regions == lbl
        area_pixels = np.sum(mask)
        area_km2 = area_pixels * grid_cell_area_km2

        mcs_id[mask] = next_cluster_id
        lifetime_dict[next_cluster_id] = 1
        max_area_dict[next_cluster_id] = area_km2
        mcs_lifetime[mask] = 1

        assigned_ids_map[lbl] = next_cluster_id
        next_cluster_id += 1

    return assigned_ids_map, next_cluster_id


def handle_continuation(
    new_label,
    old_track_id,
    final_labeled_regions,
    mcs_id,
    mcs_lifetime,
    lifetime_dict,
    max_area_dict,
    grid_cell_area_km2,
):
    """
    Continues an existing old_track_id for the new_label cluster.

    Args:
        new_label (int): Label in the current detection.
        old_track_id (int): Old track ID to be continued.
        final_labeled_regions (numpy.ndarray): Current labeled regions.
        mcs_id (numpy.ndarray): 2D array for per-pixel track IDs.
        mcs_lifetime (numpy.ndarray): 2D array for per-pixel lifetime.
        lifetime_dict (dict): Tracks how many timesteps each track ID has existed.
        max_area_dict (dict): Track ID -> maximum area encountered.
        grid_cell_area_km2 (float): Factor to convert pixel count to km².
    """
    mask = final_labeled_regions == new_label
    area_pixels = np.sum(mask)
    area_km2 = area_pixels * grid_cell_area_km2

    mcs_id[mask] = old_track_id
    lifetime_dict[old_track_id] += 1
    mcs_lifetime[mask] = lifetime_dict[old_track_id]
    if area_km2 > max_area_dict[old_track_id]:
        max_area_dict[old_track_id] = area_km2


def compute_max_consecutive(bool_list):
    """
    Compute the maximum number of consecutive True values in a Boolean list.

    Args:
        bool_list (List[bool]): List of Boolean values.

    Returns:
        int: Maximum consecutive True values.
    """
    max_cons = 0
    current = 0
    for b in bool_list:
        if b:
            current += 1
            if current > max_cons:
                max_cons = current
        else:
            current = 0
    return max_cons


def apply_robust_mask(arr, robust_flag_dict):
    """
    Vectorized function that returns track id if robust_flag_dict[tid] is True, else 0.

    Args:
        arr (np.ndarray): 2D array of track IDs.
        robust_flag_dict (dict): Dictionary mapping track id to robust flag (bool).

    Returns:
        np.ndarray: Masked array with only robust track IDs.
    """
    vec_lookup = np.vectorize(
        lambda tid: tid if robust_flag_dict.get(tid, False) else 0
    )
    return vec_lookup(arr)


def build_tracking_centers(previous_cluster_ids, center_points_dict):
    """
    Build a dictionary mapping track id to center coordinates using the provided center_points.

    Args:
        previous_cluster_ids (dict): Mapping from detection label to track id.
        center_points_dict (dict): Mapping from detection label (as string) to (lat, lon) tuple.

    Returns:
        dict: Mapping from track id (as string) to (center_lat, center_lon).
    """
    centers = {}
    for lbl, tid in previous_cluster_ids.items():
        tid_str = str(tid)
        if tid_str in center_points_dict:
            centers[tid_str] = center_points_dict[tid_str]
    return centers
