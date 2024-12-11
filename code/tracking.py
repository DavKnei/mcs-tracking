import numpy as np
import os
import warnings
import datetime
import xarray as xr
from dataclasses import dataclass
from typing import List
from collections import defaultdict

@dataclass
class MergingEvent:
    time: datetime.datetime
    parent_ids: List[int]
    child_id: int
    parent_areas: List[float]
    child_area: float

@dataclass
class SplittingEvent:
    time: datetime.datetime
    parent_id: int
    child_ids: List[int]
    parent_area: float
    child_areas: List[float]

def assign_new_id(label, cluster_mask, area, next_cluster_id, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime):
    """
    Assign a new ID to a cluster that does not overlap with any previous cluster.

    Parameters:
    - label: Current cluster label from detection (for reference).
    - cluster_mask: Boolean mask for the current cluster.
    - area: Area of the current cluster.
    - next_cluster_id: The next available ID to assign.
    - lifetime_dict: Dict tracking lifetimes of clusters by ID.
    - max_area_dict: Dict tracking max area of clusters by ID.
    - mcs_id: 2D array for assigning IDs.
    - mcs_lifetime: 2D array for lifetime per pixel.

    Returns:
    - assigned_id: The new assigned cluster ID.
    - next_cluster_id: Updated next_cluster_id.
    """
    mcs_id[cluster_mask] = next_cluster_id
    lifetime_dict[next_cluster_id] = 1
    max_area_dict[next_cluster_id] = area
    mcs_lifetime[cluster_mask] = 1
    assigned_id = next_cluster_id
    next_cluster_id += 1
    return assigned_id, next_cluster_id

def assign_ids_based_on_overlap(
    previous_labeled_regions,
    final_labeled_regions,
    previous_cluster_ids,
    next_cluster_id,
    lifetime_dict,
    max_area_dict,
    mcs_id,
    mcs_lifetime,
    overlap_threshold=10,
):
    """
    Assign consistent cluster IDs based on spatial overlap with previous timestep.

    Parameters:
    - previous_labeled_regions: 2D array of cluster labels from previous timestep.
    - final_labeled_regions: 2D array of cluster labels from current timestep.
    - previous_cluster_ids: Dict mapping previous cluster labels to their IDs.
    - next_cluster_id: The next available cluster ID for new clusters.
    - lifetime_dict: Dict tracking the lifetime of each cluster by ID.
    - max_area_dict: Dict tracking max area of each cluster by ID.
    - mcs_id: 2D array for assigning IDs this timestep.
    - mcs_lifetime: 2D array for lifetime per pixel this timestep.
    - overlap_threshold: Minimum overlap percentage to consider a match.

    Returns:
    - current_cluster_ids: Dict mapping current cluster labels to assigned IDs.
    - next_cluster_id: Updated next_cluster_id
    """
    current_cluster_ids = {}

    unique_prev_labels = np.unique(previous_labeled_regions)
    unique_prev_labels = unique_prev_labels[unique_prev_labels != -1]
    unique_curr_labels = np.unique(final_labeled_regions)
    unique_curr_labels = unique_curr_labels[unique_curr_labels != -1]

    for curr_label in unique_curr_labels:
        curr_mask = final_labeled_regions == curr_label
        curr_area = np.sum(curr_mask)
        overlap_areas = {}

        for prev_label in unique_prev_labels:
            prev_mask = previous_labeled_regions == prev_label
            overlap = np.logical_and(curr_mask, prev_mask)
            overlap_area = np.sum(overlap)
            if overlap_area > 0:
                overlap_percentage = (overlap_area / curr_area) * 100
                if overlap_percentage >= overlap_threshold:
                    overlap_areas[prev_label] = overlap_percentage

        if len(overlap_areas) == 0:
            # No suitable overlap, assign new ID
            assigned_id, next_cluster_id = assign_new_id(curr_label, curr_mask, curr_area, next_cluster_id, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime)
            current_cluster_ids[curr_label] = assigned_id
        else:
            # Continue from best overlap
            best_prev_label = max(overlap_areas, key=overlap_areas.get)
            assigned_id = previous_cluster_ids[best_prev_label]

            # Use handle_continuation logic:
            handle_continuation(curr_label, curr_mask, assigned_id, curr_area, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime)
            current_cluster_ids[curr_label] = assigned_id

    return current_cluster_ids, next_cluster_id


def handle_continuation(label, cluster_mask, assigned_id, area, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime):
    """
    Handle the continuation case when exactly one previous cluster overlaps sufficiently
    with the current cluster. The current cluster continues the previous cluster's ID.

    Parameters:
    - label: Current cluster label
    - cluster_mask: Boolean mask of the current cluster
    - assigned_id: The ID of the overlapping previous cluster
    - area: Area of the current cluster (km²)
    - lifetime_dict: Dict tracking the lifetime of each ID
    - max_area_dict: Dict tracking the maximum area for each ID
    - mcs_id: 2D array to assign MCS IDs for this timestep
    - mcs_lifetime: 2D array to assign lifetime values for this timestep
    """
    mcs_id[cluster_mask] = assigned_id
    lifetime_dict[assigned_id] += 1
    mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
    if area > max_area_dict[assigned_id]:
        max_area_dict[assigned_id] = area
    

def get_dominant_cluster(prev_ids, max_area_dict):
    """Find the dominant cluster (largest area) among prev_ids
    
    Parameters:
    - prev_ids: List of previous cluster IDs
    - max_area_dict: Dict mapping cluster IDs to their max area
    
    Returns:
    - best_id: The ID of the dominant cluster
    """
    best_id = None
    best_area = -1
    for pid in prev_ids:
        if max_area_dict.get(pid,0) > best_area:
            best_area = max_area_dict[pid]
            best_id = pid
    return best_id

def handle_merging(label, cluster_mask, prev_ids, area, nmaxmerge, current_time, lifetime_dict, max_area_dict, mcs_id, merging_events, mcs_lifetime):
    """
    Handle merging events where multiple previous IDs overlap with one current cluster.
    Keeps the dominant parent's ID and records a MergingEvent.

    Parameters:
    - label: Current cluster label
    - cluster_mask: Boolean mask for this cluster
    - prev_ids: List of previous cluster IDs involved in merging
    - area: Area of current cluster (km²)
    - nmaxmerge: Maximum allowed merging clusters
    - current_time: Current timestamp
    - lifetime_dict, max_area_dict: Dictionaries tracking lifetime and max area
    - mcs_id: 2D array for MCS IDs
    - merging_events: List to append MergingEvent
    - mcs_lifetime: 2D array for lifetime this timestep

    Returns:
    - assigned_id: The ID chosen for the merged cluster (dominant parent's ID)
    """
    dominant_parent_id = get_dominant_cluster(prev_ids, max_area_dict)
    assigned_id = dominant_parent_id

    handle_continuation(label, cluster_mask, assigned_id, area, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime)

    parent_areas = [max_area_dict[pid] for pid in prev_ids]
    if len(prev_ids) > nmaxmerge:
        prev_ids = prev_ids[:nmaxmerge]
        parent_areas = parent_areas[:nmaxmerge]
        warnings.warn(
            f"Number of merging clusters exceeds nmaxmerge ({nmaxmerge}) at time {current_time}."
        )
    merge_event = MergingEvent(
        time=current_time,
        parent_ids=prev_ids,
        child_id=assigned_id,
        parent_areas=parent_areas,
        child_area=area,
    )
    merging_events.append(merge_event)
    return assigned_id

def handle_splitting_final_step(overlaps_with_curr, current_cluster_ids, max_area_dict, lifetime_dict, next_cluster_id, nmaxmerge, current_time, splitting_events):
    """
    Handle the final splitting logic after all clusters in this timestep are processed.
    If a previous ID overlaps multiple current clusters, we identify it as splitting.

    We ensure the largest child keeps the parent_id and assign new IDs to the others.

    Parameters:
    - overlaps_with_curr: Dict mapping prev_id -> list of current labels
    - current_cluster_ids: Dict mapping current cluster labels to their assigned IDs
    - max_area_dict, lifetime_dict: Tracking area and lifetime
    - next_cluster_id: Next available ID for new clusters
    - nmaxmerge: Max allowed merging/splitting
    - current_time: Current timestamp
    - splitting_events: List to append SplittingEvent

    Returns:
    - next_cluster_id: Updated next_cluster_id after assigning new IDs
    """
    for parent_id, curr_labels in overlaps_with_curr.items():
        if len(curr_labels) > 1:
            # Splitting event
            child_ids = []
            child_areas = []
            for clab in curr_labels:
                cid = current_cluster_ids[clab]
                child_ids.append(cid)
                child_areas.append(max_area_dict[cid])
            parent_area = max_area_dict[parent_id]

            largest_idx = np.argmax(child_areas)
            largest_child_id = child_ids[largest_idx]

            if largest_child_id != parent_id:
                # If parent_id is among child_ids:
                if parent_id in child_ids:
                    # Assign new IDs to all non-largest children
                    for i, cid in enumerate(child_ids):
                        if i != largest_idx:
                            child_ids[i] = next_cluster_id
                            lifetime_dict[next_cluster_id] = 1
                            max_area_dict[next_cluster_id] = child_areas[i]
                            next_cluster_id += 1
                    # Update parent's area/lifetime if needed
                    # largest_child_id had some lifetime/area
                    # parent_id remains parent_id
                    lifetime_dict[parent_id] = lifetime_dict[largest_child_id]
                    max_area_dict[parent_id] = child_areas[largest_idx]
                else:
                    # parent_id not in child_ids: just assign new IDs to others
                    for i, cid in enumerate(child_ids):
                        if i != largest_idx:
                            child_ids[i] = next_cluster_id
                            lifetime_dict[next_cluster_id] = 1
                            max_area_dict[next_cluster_id] = child_areas[i]
                            next_cluster_id += 1
            else:
                # largest child already has parent_id, assign new IDs to others
                for i, cid in enumerate(child_ids):
                    if i != largest_idx:
                        child_ids[i] = next_cluster_id
                        lifetime_dict[next_cluster_id] = 1
                        max_area_dict[next_cluster_id] = child_areas[i]
                        next_cluster_id += 1

            if len(child_ids) > nmaxmerge:
                child_ids = child_ids[:nmaxmerge]
                child_areas = child_areas[:nmaxmerge]
                warnings.warn(
                    f"Number of splitting clusters exceeds nmaxmerge ({nmaxmerge}) at time {current_time}."
                )

            split_event = SplittingEvent(
                time=current_time,
                parent_id=parent_id,
                child_ids=child_ids,
                parent_area=parent_area,
                child_areas=child_areas,
            )
            splitting_events.append(split_event)

    return next_cluster_id

def filter_main_mcs(mcs_id_list, main_mcs_ids):
    """
    Filter tracking results to include only main MCS tracks.

    Parameters:
    - mcs_id_list: List of MCS ID arrays.
    - main_mcs_ids: List of MCS IDs identified as main MCS.

    Returns:
    - filtered_mcs_id_list: List of MCS ID arrays with only main MCS IDs.
    """
    filtered_mcs_id_list = []
    for mcs_id_array in mcs_id_list:
        filtered_array = mcs_id_array.copy()
        mask = ~np.isin(filtered_array, main_mcs_ids)
        filtered_array[mask] = 0
        filtered_mcs_id_list.append(filtered_array)
    return filtered_mcs_id_list

def track_mcs(
    detection_results,
    main_lifetime_thresh,
    main_area_thresh,
    grid_cell_area_km2,
    nmaxmerge,
):
    """
    Track MCSs across multiple timesteps using spatial overlap and stable ID assignment.

    Parameters:
    - detection_results: List of dicts with keys "final_labeled_regions", "time", "lat", "lon".
    - main_lifetime_thresh: Minimum lifetime for main MCS.
    - main_area_thresh: Minimum area for main MCS (km²).
    - grid_cell_area_km2: Grid cell area in km².
    - nmaxmerge: Max allowed number of clusters in a merge/split.

    Returns:
    - mcs_detected_list: List of binary arrays per timestep (MCS detected =1).
    - mcs_id_list: List of ID arrays per timestep.
    - lifetime_list: List of arrays for pixel-wise lifetime.
    - time_list: List of timestamps.
    - lat: 2D lat array from first timestep.
    - lon: 2D lon array from first timestep.
    - main_mcs_ids: List of IDs considered main MCS by end.
    - merging_events: List of MergingEvent instances.
    - splitting_events: List of SplittingEvent instances.
    """
    from collections import defaultdict

    previous_labeled_regions = None
    previous_cluster_ids = {}
    next_cluster_id = 1

    mcs_detected_list = []
    mcs_id_list = []
    lifetime_list = []
    time_list = []
    lat = None
    lon = None

    lifetime_dict = defaultdict(int)
    max_area_dict = defaultdict(float)

    merging_events = []
    splitting_events = []

    for idx, detection_result in enumerate(detection_results):
        final_labeled_regions = detection_result["final_labeled_regions"]
        current_time = detection_result["time"]
        current_lat = detection_result["lat"]
        current_lon = detection_result["lon"]

        if lat is None:
            lat = current_lat
            lon = current_lon

        mcs_detected = np.zeros_like(final_labeled_regions, dtype=np.int8)
        mcs_id = np.zeros_like(final_labeled_regions, dtype=np.int32)
        mcs_lifetime = np.zeros_like(final_labeled_regions, dtype=np.int32)

        if previous_labeled_regions is None:
            # First timestep, assign new IDs to all clusters
            unique_labels = np.unique(final_labeled_regions)
            unique_labels = unique_labels[unique_labels != -1]
            for label in unique_labels:
                cluster_mask = final_labeled_regions == label
                area = np.sum(cluster_mask)*grid_cell_area_km2
                assigned_id, next_cluster_id = assign_new_id(label, cluster_mask, area, next_cluster_id, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime)
                previous_cluster_ids[label] = assigned_id
                mcs_detected[cluster_mask] = 1
        else:
            # Subsequent timesteps: assign IDs based on overlap
            current_cluster_ids, next_cluster_id = assign_ids_based_on_overlap(
                previous_labeled_regions, final_labeled_regions, previous_cluster_ids,
                next_cluster_id, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime,
                overlap_threshold=10
            )

            # Compute overlaps_with_curr for merging/splitting detection if needed
            overlaps_with_curr = defaultdict(list)
            # Populate overlaps_with_curr by comparing prev_id to current clusters:
            # This step depends on how you detect merging/splitting.
            # If merging/splitting logic requires multiple previous or current clusters,
            # you can compute it similarly to before:
            unique_prev_labels = np.unique(previous_labeled_regions)
            unique_prev_labels = unique_prev_labels[unique_prev_labels != -1]
            unique_curr_labels = np.unique(final_labeled_regions)
            unique_curr_labels = unique_curr_labels[unique_curr_labels != -1]

            for prev_label, prev_id in previous_cluster_ids.items():
                prev_mask = (previous_labeled_regions == prev_label)
                curr_matches = []
                for curr_label in unique_curr_labels:
                    curr_mask = (final_labeled_regions == curr_label)
                    overlap = np.logical_and(prev_mask, curr_mask)
                    overlap_cells = np.sum(overlap)
                    if overlap_cells > 0:
                        curr_area = np.sum(curr_mask)
                        overlap_percentage = (overlap_cells / curr_area)*100
                        if overlap_percentage >= 10:
                            curr_matches.append(curr_label)
                if len(curr_matches) > 0:
                    overlaps_with_curr[prev_id] = curr_matches

            # Check if splitting occurred
            if any(len(vals) > 1 for vals in overlaps_with_curr.values()):
                next_cluster_id = handle_splitting_final_step(
                    overlaps_with_curr, current_cluster_ids, max_area_dict, lifetime_dict,
                    next_cluster_id, nmaxmerge, current_time, splitting_events
                )

            # Assign mcs_detected =1 for all clusters
            unique_labels = np.unique(final_labeled_regions)
            unique_labels = unique_labels[unique_labels != -1]
            for label in unique_labels:
                cluster_mask = final_labeled_regions == label
                mcs_detected[cluster_mask] = 1

            previous_cluster_ids = current_cluster_ids

        previous_labeled_regions = final_labeled_regions.copy()

        # Append results
        mcs_detected_list.append(mcs_detected)
        mcs_id_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        total_lifetime_dict = lifetime_dict
        main_mcs_ids = [
            uid for uid in total_lifetime_dict.keys()
            if total_lifetime_dict[uid] >= main_lifetime_thresh and max_area_dict.get(uid,0) >= main_area_thresh
        ]

    return (
        mcs_detected_list,
        mcs_id_list,
        lifetime_list,
        time_list,
        lat,
        lon,
        main_mcs_ids,
        merging_events,
        splitting_events,
    )
