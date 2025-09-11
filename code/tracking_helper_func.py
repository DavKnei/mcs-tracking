import numpy as np
from scipy.ndimage import affine_transform, center_of_mass, label

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

def attempt_advection_rescue(
    labels_no_overlap,
    previous_labeled_regions,
    final_labeled_regions,
    previous_cluster_ids,
    grid_cell_area_km2,
    overlap_threshold=10, 
    search_radius_km=1000,
):
    """
    Attempts to rescue non-overlapping systems by applying an advection correction.

    For each system that has no initial overlap, this function calculates a
    local displacement vector based on the movement of nearby, successfully
    tracked systems. It then displaces all systems from the previous timestep
    by this vector and re-checks for overlaps.

    Args:
        labels_no_overlap (list): A list of labels from the current timestep
                                  that did not have any spatial overlap.
        previous_labeled_regions (np.ndarray): The labeled mask from the previous timestep (t-1).
        final_labeled_regions (np.ndarray): The labeled mask from the current timestep (t).
        previous_cluster_ids (dict): Mapping of {label_t-1: track_id}.
        grid_cell_area_km2 (float): The area of a single grid cell in km².
        overlap_threshold (float): Minimum percentage overlap required for consideration.
        search_radius_km (int): The radius in km to search for neighboring systems.

    Returns:
        dict: A dictionary of newly found overlaps in the format {new_label: [old_track_id]}.
    """
    if not labels_no_overlap:
        return {}

    rescued_overlaps = {}
    radius_pixels = search_radius_km / np.sqrt(grid_cell_area_km2)

    # Pre-calculate centers for all labels to avoid redundant calculations
    current_centers = {lbl: center_of_mass(final_labeled_regions, final_labeled_regions, lbl) for lbl in np.unique(final_labeled_regions) if lbl != 0}
    previous_centers = {lbl: center_of_mass(previous_labeled_regions, previous_labeled_regions, lbl) for lbl in np.unique(previous_labeled_regions) if lbl != 0}

    # Pre-calculate areas for previous labels
    previous_areas = {lbl: np.sum(previous_labeled_regions == lbl) for lbl in previous_centers.keys()}

    for new_lbl in labels_no_overlap:
        center_new = current_centers.get(new_lbl)
        if not center_new:
            continue

        displacement_vectors = []
        # Find stable, one-to-one tracked neighbors within the radius
        for prev_lbl, prev_center in previous_centers.items():
            if abs(prev_center[0] - center_new[0]) < radius_pixels and abs(prev_center[1] - center_new[1]) < radius_pixels:
                overlapping_mask = final_labeled_regions[previous_labeled_regions == prev_lbl]
                unique_overlapping_labels = np.unique(overlapping_mask[overlapping_mask != 0])
                if len(unique_overlapping_labels) == 1:
                    partner_new_lbl = unique_overlapping_labels[0]
                    center_partner_new = current_centers.get(partner_new_lbl)
                    if center_partner_new:
                        dy = center_partner_new[0] - prev_center[0]
                        dx = center_partner_new[1] - prev_center[1]
                        displacement_vectors.append((dy, dx))

        if not displacement_vectors:
            continue

        avg_dy, avg_dx = np.mean(displacement_vectors, axis=0)

        # Displace the previous mask
        translated_previous_regions = np.zeros_like(previous_labeled_regions)
        transform_matrix = np.array([[1, 0, -avg_dy], [0, 1, -avg_dx], [0, 0, 1]])
        affine_transform(previous_labeled_regions, transform_matrix, output=translated_previous_regions, order=0)

        current_mask = final_labeled_regions == new_lbl
        area_new = np.sum(current_mask)
        
        # Find all potentially overlapping old labels after translation
        overlapping_old_labels = np.unique(translated_previous_regions[current_mask])
        overlapping_old_labels = overlapping_old_labels[overlapping_old_labels != 0]

        if len(overlapping_old_labels) > 0:
            potential_matches = []
            for old_lbl in overlapping_old_labels:
                area_old = previous_areas.get(old_lbl, 0)
                if area_old == 0:
                    continue

                # Calculate intersection area
                intersection_mask = (current_mask) & (translated_previous_regions == old_lbl)
                intersection_area = np.sum(intersection_mask)
                
                # Check if overlap percentage is sufficient
                if (intersection_area / min(area_new, area_old)) * 100 >= overlap_threshold:
                    if old_lbl in previous_cluster_ids:
                        potential_matches.append(previous_cluster_ids[old_lbl])

            if potential_matches:
                rescued_overlaps[new_lbl] = potential_matches
            
    return rescued_overlaps

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
