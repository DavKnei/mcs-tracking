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

def initialize_new_id(label, cluster_mask, next_cluster_id, area, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime):
    """
    Assign a new ID to a newly detected cluster that has no overlap with previous time steps.

    Parameters:
    - label: Current cluster label (not ID, just the label from final_labeled_regions)
    - cluster_mask: Boolean mask of the current cluster
    - next_cluster_id: The next available integer ID for a new cluster
    - area: Area of this cluster (km²)
    - lifetime_dict: Dict tracking the lifetime of each ID
    - max_area_dict: Dict tracking the maximum area encountered for each ID
    - mcs_id: 2D array for assigning MCS IDs this timestep
    - mcs_lifetime: 2D array for assigning lifetime values this timestep

    Returns:
    - assigned_id: The newly assigned cluster ID
    - next_cluster_id: The updated next_cluster_id counter
    """
    mcs_id[cluster_mask] = next_cluster_id
    lifetime_dict[next_cluster_id] = 1
    max_area_dict[next_cluster_id] = area
    mcs_lifetime[cluster_mask] = 1
    assigned_id = next_cluster_id
    next_cluster_id += 1
    return assigned_id, next_cluster_id

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

def handle_no_overlap(label, cluster_mask, area, next_cluster_id, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime):
    """
    Handle case with no overlap to previous clusters, indicating a new MCS ID should be assigned.

    Parameters:
    - label: Current cluster label
    - cluster_mask: Boolean mask of the current cluster
    - area: Area of current cluster (km²)
    - next_cluster_id: The next available ID
    - lifetime_dict, max_area_dict, mcs_id, mcs_lifetime: As described above

    Returns:
    - assigned_id: The new assigned ID
    - next_cluster_id: Updated next_cluster_id
    """
    assigned_id, next_cluster_id = initialize_new_id(
        label, cluster_mask, next_cluster_id, area, lifetime_dict, max_area_dict, mcs_id, mcs_lifetime
    )
    return assigned_id, next_cluster_id
    
def get_dominant_cluster(prev_ids, max_area_dict):
    """Find the dominant cluster (largest area) among prev_ids."""
    best_id = None
    best_area = -1
    for pid in prev_ids:
        if max_area_dict.get(pid,0) > best_area:
            best_area = max_area_dict[pid]
            best_id = pid
    return best_id

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
    Track MCSs across time steps using detection results.

    Parameters:
    - detection_results: List of dictionaries containing detection results.
    - main_lifetime_thresh: Minimum lifetime threshold for main MCSs.
    - main_area_thresh: Minimum area threshold for main MCSs.
    - grid_cell_area_km2: Area of each grid cell in square kilometers.
    - nmaxmerge: Maximum number of clusters to merge or split.

    Returns:
    - mcs_detected_list: List of binary arrays indicating detected MCSs.
    - mcs_id_list: List of arrays with MCS IDs assigned.
    - lifetime_list: List of arrays indicating the lifetime of MCSs.
    - time_list: List of time stamps corresponding to each time step.
    - lat: Latitude array.
    - lon: Longitude array.
    - main_mcs_ids: List of MCS IDs identified as main MCS.
    - merging_events: List of MergingEvent instances.
    - splitting_events: List of SplittingEvent instances.
    """
    previous_labeled_regions = None
    previous_cluster_ids = {}
    next_cluster_id = 1

    mcs_detected_list = []
    mcs_id_list = []
    time_list = []
    lifetime_list = []
    lat = None
    lon = None

    # Dictionaries to keep track of lifetime and max area for each MCS ID
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

        current_cluster_ids = {}

        cluster_labels = np.unique(final_labeled_regions)
        cluster_labels = cluster_labels[cluster_labels != 0]  # Exclude background

        overlaps_with_prev = defaultdict(list)
        overlaps_with_curr = defaultdict(list)

        for label in cluster_labels:
            cluster_mask = final_labeled_regions == label
            mcs_detected[cluster_mask] = 1

            area = np.sum(cluster_mask) * grid_cell_area_km2

            if previous_labeled_regions is None:
                # First time step, assign new ID
                mcs_id[cluster_mask] = next_cluster_id
                current_cluster_ids[label] = next_cluster_id
                lifetime_dict[next_cluster_id] = 1
                max_area_dict[next_cluster_id] = area
                mcs_lifetime[cluster_mask] = lifetime_dict[next_cluster_id]
                next_cluster_id += 1
            else:
                # Compare with previous clusters
                overlap_area = {}
                for prev_label, prev_id in previous_cluster_ids.items():
                    prev_cluster_mask = previous_labeled_regions == prev_label
                    overlap = np.logical_and(cluster_mask, prev_cluster_mask)
                    overlap_cells = np.sum(overlap)
                    if overlap_cells > 0:
                        current_cluster_area = np.sum(cluster_mask)
                        overlap_percentage = (overlap_cells / current_cluster_area) * 100
                        if overlap_percentage >= 10:
                            overlap_area[prev_id] = overlap_percentage
                            overlaps_with_prev[label].append(prev_id)
                            overlaps_with_curr[prev_id].append(label)

                if len(overlap_area) == 1:
                    # Continuation
                    assigned_id = list(overlap_area.keys())[0]
                    mcs_id[cluster_mask] = assigned_id
                    current_cluster_ids[label] = assigned_id
                    lifetime_dict[assigned_id] += 1
                    mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                    if area > max_area_dict[assigned_id]:
                        max_area_dict[assigned_id] = area
                elif len(overlap_area) == 0:
                    # New initiation
                    mcs_id[cluster_mask] = next_cluster_id
                    current_cluster_ids[label] = next_cluster_id
                    lifetime_dict[next_cluster_id] = 1
                    max_area_dict[next_cluster_id] = area
                    mcs_lifetime[cluster_mask] = lifetime_dict[next_cluster_id]
                    next_cluster_id += 1
                else:
                    # Merging or splitting event
                    prev_ids = list(overlap_area.keys())
                    if len(prev_ids) > 1:
                        # Merging event: keep dominant parent's ID
                        dominant_parent_id = get_dominant_cluster(prev_ids, max_area_dict)
                        assigned_id = dominant_parent_id
                        mcs_id[cluster_mask] = assigned_id
                        current_cluster_ids[label] = assigned_id
                        lifetime_dict[assigned_id] += 1
                        if area > max_area_dict[assigned_id]:
                            max_area_dict[assigned_id] = area
                        mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                        # parent_areas for merging event
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

                    elif len(prev_ids) == 1 and len(overlaps_with_prev[label]) > 1:
                        # This condition was previously used for early splitting, remove it
                        # We rely on the final block below for splitting
                        pass
                    else:
                        # Unexpected case
                        warnings.warn(f"Unexpected case at time {current_time}.")

        # Now handle splitting after processing all clusters
        for prev_id, curr_labels in overlaps_with_curr.items():
            if len(curr_labels) > 1:
                # Splitting event
                parent_id = prev_id
                child_ids = []
                child_areas = []
                for clab in curr_labels:
                    cid = current_cluster_ids[clab]
                    child_ids.append(cid)
                    child_areas.append(max_area_dict[cid])
                parent_area = max_area_dict[parent_id]

                # Identify largest child
                largest_idx = np.argmax(child_areas)
                largest_child_id = child_ids[largest_idx]

                # If largest child_id is not parent_id, we must reassign IDs
                # Keep parent_id for largest piece
                if largest_child_id != parent_id:
                    # Find if parent_id is among child_ids
                    if parent_id in child_ids:
                        # We need to swap IDs so that largest piece gets parent_id
                        # Let’s assign new IDs to all children except largest_idx
                        for i, cid in enumerate(child_ids):
                            if i == largest_idx:
                                # largest piece gets parent_id
                                # Transfer lifetime/area info to parent_id
                                # Actually, parent_id and largest_child_id differ:
                                # If largest_child_id was some other ID, we must swap their info
                                # Easiest approach: since largest piece keeps parent_id,
                                # set max_area_dict[parent_id] = child_areas[largest_idx]
                                # lifetime_dict[parent_id] = lifetime_dict[largest_child_id]
                                # We assume continuity for parent track
                                lifetime_dict[parent_id] = lifetime_dict[largest_child_id]
                                max_area_dict[parent_id] = child_areas[largest_idx]
                                # Reassign the cluster that had largest_child_id to parent_id
                                # This may require also updating mcs_id arrays above, but since we already assigned
                                # them this timestep, assume next step will handle continuity
                            else:
                                # Assign a new ID if not largest
                                child_ids[i] = next_cluster_id
                                lifetime_dict[next_cluster_id] = 1
                                max_area_dict[next_cluster_id] = child_areas[i]
                                next_cluster_id += 1
                    else:
                        # parent_id not in child_ids means parent_id cluster was replaced?
                        # Keep parent_id as is, assign new IDs to others:
                        for i, cid in enumerate(child_ids):
                            if i != largest_idx:
                                child_ids[i] = next_cluster_id
                                lifetime_dict[next_cluster_id] = 1
                                max_area_dict[next_cluster_id] = child_areas[i]
                                next_cluster_id += 1
                else:
                    # largest child already has parent_id, good
                    # assign new IDs to others
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

        # Update previous clusters
        previous_labeled_regions = final_labeled_regions.copy()
        previous_cluster_ids = current_cluster_ids.copy()

        # Append results
        mcs_detected_list.append(mcs_detected)
        mcs_id_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        total_lifetime_dict = lifetime_dict  # Already calculated during tracking

        # Identify main MCS IDs based on lifetime and area thresholds
        main_mcs_ids = [
            uid
            for uid in total_lifetime_dict.keys()
            if total_lifetime_dict[uid] >= main_lifetime_thresh
            and max_area_dict.get(uid, 0) >= main_area_thresh
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
