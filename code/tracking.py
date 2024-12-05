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


def handle_splitting_event(overlap_area, next_cluster_id):
    """
    Handle splitting events where one previous cluster overlaps with multiple current clusters.

    Parameters:
    - overlap_area: Dictionary mapping current cluster labels to overlap percentages.
    - next_cluster_id: Integer representing the next available cluster ID.

    Returns:
    - current_cluster_ids: Dictionary mapping current cluster labels to assigned cluster IDs.
    - next_cluster_id: Updated next_cluster_id after assigning new IDs.
    """
    current_cluster_ids = {}
    for curr_label in overlap_area.keys():
        # Assign new IDs to the split clusters
        current_cluster_ids[curr_label] = next_cluster_id
        next_cluster_id += 1
    return current_cluster_ids, next_cluster_id


def handle_merging_event(overlap_area, previous_cluster_ids):
    """
    Handle merging events where multiple previous clusters overlap with one current cluster.

    Parameters:
    - overlap_area: Dictionary mapping previous cluster IDs to overlap percentages.
    - previous_cluster_ids: Dictionary mapping previous cluster labels to cluster IDs.

    Returns:
    - assigned_id: Cluster ID assigned to the merged cluster.
    """
    # For merging, you might choose to assign a new ID or keep one of the existing IDs
    # Here, we assign a new unique ID
    assigned_id = min(overlap_area.keys())  # Or use next_cluster_id to assign a new ID
    return assigned_id


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
        # Create a copy to avoid modifying the original array
        filtered_array = mcs_id_array.copy()
        # Set IDs not in main_mcs_ids to zero
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
    - main_mcs_ids: List of MCS IDs identified as main MCSs.
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

    # Dictionary to keep track of the lifetime for each MCS ID
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

        # Mapping from current cluster labels to overlapping previous IDs
        overlaps_with_prev = defaultdict(list)

        # Mapping from previous cluster IDs to overlapping current labels
        overlaps_with_curr = defaultdict(list)

        for label in cluster_labels:
            cluster_mask = final_labeled_regions == label
            mcs_detected[cluster_mask] = 1

            # Calculate area
            area = np.sum(cluster_mask) * grid_cell_area_km2

            if previous_labeled_regions is None:
                # First time step, assign new IDs
                mcs_id[cluster_mask] = next_cluster_id
                current_cluster_ids[label] = next_cluster_id
                # Initialize lifetime
                lifetime_dict[next_cluster_id] = 1
                # Calculate area
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
                        # Calculate overlap percentage relative to current cluster
                        current_cluster_area = np.sum(cluster_mask)
                        overlap_percentage = (
                            overlap_cells / current_cluster_area
                        ) * 100
                        if overlap_percentage >= 10:
                            overlap_area[prev_id] = overlap_percentage
                            overlaps_with_prev[label].append(prev_id)
                            overlaps_with_curr[prev_id].append(label)

                if len(overlap_area) == 1:
                    # Continuation, assign existing ID
                    assigned_id = list(overlap_area.keys())[0]
                    mcs_id[cluster_mask] = assigned_id
                    current_cluster_ids[label] = assigned_id
                    # Update lifetime
                    lifetime_dict[assigned_id] += 1
                    mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                    # Update max area
                    area = np.sum(cluster_mask) * grid_cell_area_km2
                    if area > max_area_dict[assigned_id]:
                        max_area_dict[assigned_id] = area
                elif len(overlap_area) == 0:
                    # New initation, assign new ID and lifetime
                    mcs_id[cluster_mask] = next_cluster_id
                    current_cluster_ids[label] = next_cluster_id
                    # Initialize lifetime
                    lifetime_dict[next_cluster_id] = 1
                    max_area_dict[next_cluster_id] = area
                    mcs_lifetime[cluster_mask] = lifetime_dict[next_cluster_id]
                    next_cluster_id += 1
                else:
                    # Merging or splitting event
                    prev_ids = list(overlap_area.keys())
                    if len(prev_ids) > 1:
                        # Merging event
                        assigned_id = next_cluster_id
                        next_cluster_id += 1
                        mcs_id[cluster_mask] = assigned_id
                        current_cluster_ids[label] = assigned_id
                        # Initialize lifetime
                        lifetime_dict[assigned_id] = 1
                        max_area_dict[assigned_id] = area
                        mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                        # Get areas of parent tracks
                        parent_areas = [max_area_dict[pid] for pid in prev_ids]
                        # Limit number of mergers
                        if len(prev_ids) > nmaxmerge:
                            # Trim the list and issue a warning
                            prev_ids = prev_ids[:nmaxmerge]
                            parent_areas = parent_areas[:nmaxmerge]
                            warnings.warn(
                                f"Number of merging clusters exceeds nmaxmerge ({nmaxmerge}) at time {current_time}."
                            )
                        # Record merging event
                        merge_event = MergingEvent(
                            time=current_time,
                            parent_ids=prev_ids,
                            child_id=assigned_id,
                            parent_areas=parent_areas,
                            child_area=area,
                        )
                        merging_events.append(merge_event)

                    elif len(prev_ids) == 1 and len(overlaps_with_prev[label]) > 1:
                        # Splitting event
                        parent_id = prev_ids[0]
                        assigned_id = next_cluster_id
                        next_cluster_id += 1
                        mcs_id[cluster_mask] = assigned_id
                        current_cluster_ids[label] = assigned_id
                        # Initialize lifetime
                        lifetime_dict[assigned_id] = 1
                        max_area_dict[assigned_id] = area
                        mcs_lifetime[cluster_mask] = lifetime_dict[assigned_id]
                        # Get area of parent track
                        parent_area = max_area_dict[parent_id]
                        # Record splitting event
                        split_event = SplittingEvent(
                            time=current_time,
                            parent_id=parent_id,
                            child_ids=[assigned_id],
                            parent_area=parent_area,
                            child_areas=[area],
                        )
                        splitting_events.append(split_event)
                    else:
                        # Should not reach here
                        warnings.warn(f"Unexpected case at time {current_time}.")

        for prev_id, curr_labels in overlaps_with_curr.items():
            if len(curr_labels) > 1:
                # Splitting event detected
                parent_id = prev_id
                child_ids = []
                child_areas = []
                for label in curr_labels:
                    assigned_id = current_cluster_ids[label]
                    child_ids.append(assigned_id)
                    child_areas.append(max_area_dict[assigned_id])
                parent_area = max_area_dict[parent_id]
                # Limit number of splitters
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
