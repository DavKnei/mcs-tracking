import numpy as np
import datetime
import os
import warnings
import datetime
import xarray as xr
from dataclasses import dataclass
from typing import List
from collections import defaultdict


@dataclass
class MergingEvent:
    """Stores information about a merging event in the tracking."""

    time: datetime.datetime
    parent_ids: List[int]
    child_id: int
    parent_areas: List[float]
    child_area: float


@dataclass
class SplittingEvent:
    """Stores information about a splitting event in the tracking."""

    time: datetime.datetime
    parent_id: int
    child_ids: List[int]
    parent_area: float
    child_areas: List[float]


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


def handle_merging(
    new_label,
    old_track_ids,
    merging_events,
    final_labeled_regions,
    current_time,
    max_area_dict,
    grid_cell_area_km2,
    nmaxmerge=5,
):
    """
    Resolves a merging scenario where multiple old track IDs map to a single new label.

    Args:
        new_label (int): The new detection label that overlaps multiple old track IDs.
        old_track_ids (List[int]): The list of old track IDs that overlap this new label.
        merging_events (List[MergingEvent]): A list to record merging events.
        final_labeled_regions (numpy.ndarray): Current labeled regions.
        current_time (datetime.datetime): Timestamp for the merging event.
        max_area_dict (dict): Maps track ID -> max area encountered.
        grid_cell_area_km2 (float): Factor to convert pixel count to km².
        nmaxmerge (int, optional): Max number of merges recorded. Defaults to 5.

    Returns:
        int: The 'dominant' old track ID that this new label will inherit.
    """
    if len(old_track_ids) == 1:
        return old_track_ids[0]

    best_id = get_dominant_cluster(old_track_ids, max_area_dict)

    if len(old_track_ids) > nmaxmerge:
        old_track_ids = old_track_ids[:nmaxmerge]

    if merging_events is not None and len(old_track_ids) > 1:
        mask = final_labeled_regions == new_label
        area_pix = np.sum(mask)
        child_area = area_pix * grid_cell_area_km2
        parent_areas = [max_area_dict[pid] for pid in old_track_ids]
        mevt = MergingEvent(
            time=current_time,
            parent_ids=old_track_ids,
            child_id=best_id,
            parent_areas=parent_areas,
            child_area=child_area,
        )
        merging_events.append(mevt)

    return best_id


def handle_splitting(
    old_track_id,
    new_label_list,
    final_labeled_regions,
    current_time,
    next_cluster_id,
    splitting_events,
    mcs_id,
    mcs_lifetime,
    lifetime_dict,
    max_area_dict,
    grid_cell_area_km2,
    nmaxsplit=5,
):
    """
    Resolves splitting when one old_track_id is claimed by multiple new labels.

    Args:
        old_track_id (int): The parent track ID from the previous timestep.
        new_label_list (List[int]): List of new detection labels that overlap old_track_id.
        final_labeled_regions (numpy.ndarray): Labeled regions from current timestep.
        current_time (datetime.datetime): Timestamp for splitting event.
        next_cluster_id (int): Next available track ID for ephemeral new IDs.
        splitting_events (List[SplittingEvent]): A list to record splitting events.
        mcs_id (numpy.ndarray): 2D array for track IDs.
        mcs_lifetime (numpy.ndarray): 2D array for track lifetimes.
        lifetime_dict (dict): Maps track ID -> number of timesteps.
        max_area_dict (dict): Maps track ID -> max area encountered.
        grid_cell_area_km2 (float): Factor for pixel area.
        nmaxsplit (int, optional): Max number of splits in a single event. Defaults to 5.

    Returns:
        Tuple[dict, int]: A dictionary { new_label : final assigned track ID } and the updated next_cluster_id.
    """
    if len(new_label_list) <= 1:
        return {}, next_cluster_id  # no actual split

    new_label_areas = []
    for nlbl in new_label_list:
        mask = final_labeled_regions == nlbl
        area_pix = np.sum(mask)
        new_label_areas.append(area_pix * grid_cell_area_km2)

    idx_sorted = sorted(
        range(len(new_label_list)), key=lambda i: new_label_areas[i], reverse=True
    )
    keep_idx = idx_sorted[0]
    keep_label = new_label_list[keep_idx]
    keep_area = new_label_areas[keep_idx]

    splitted_assign_map = {}

    # The largest child keeps old_track_id
    splitted_assign_map[keep_label] = old_track_id
    lifetime_dict[old_track_id] -= 1  # Patch for double counting
    if keep_area > max_area_dict[old_track_id]:
        max_area_dict[old_track_id] = keep_area

    keep_mask = final_labeled_regions == keep_label
    mcs_id[keep_mask] = old_track_id
    mcs_lifetime[keep_mask] = lifetime_dict[old_track_id]

    splitted_child_labels = []
    splitted_child_areas = []

    for i in idx_sorted[1:]:
        lbl = new_label_list[i]
        area_s = new_label_areas[i]
        splitted_assign_map[lbl] = next_cluster_id
        lifetime_dict[next_cluster_id] = 1
        max_area_dict[next_cluster_id] = area_s
        mask_s = final_labeled_regions == lbl
        mcs_id[mask_s] = next_cluster_id
        mcs_lifetime[mask_s] = 1
        splitted_child_labels.append(next_cluster_id)
        splitted_child_areas.append(area_s)
        next_cluster_id += 1

    if splitting_events is not None and len(new_label_list) > 1:
        if len(new_label_list) > nmaxsplit:
            new_label_list = new_label_list[:nmaxsplit]
        sevt = SplittingEvent(
            time=current_time,
            parent_id=old_track_id,
            child_ids=[splitted_assign_map[lbl] for lbl in new_label_list],
            parent_area=max_area_dict[old_track_id],
            child_areas=[
                max_area_dict[splitted_assign_map[lbl]] for lbl in new_label_list
            ],
        )
        splitting_events.append(sevt)

    return splitted_assign_map, next_cluster_id


def filter_relevant_systems(
    mcs_ids_list, main_mcs_ids, merging_events, splitting_events
):
    """
    Filters the tracking results to include only relevant track IDs. Main MCSs and systems that merge into
    MCSs or split of MCSs.

    This function computes the union of:
      - main_mcs_ids (the main MCS tracks that satisfy the area–lifetime criteria),
      - All track IDs involved in merging events (both the parent IDs and the child ID), and
      - All track IDs involved in splitting events (both the parent ID and all child IDs).

    It then processes the list of MCS ID arrays (mcs_ids_list) such that any pixel value
    that is not in this union is set to 0.

    Args:
        mcs_ids_list (List[np.ndarray]): List of 2D arrays (one per timestep) with track IDs.
        main_mcs_ids (List[int]): List of track IDs that satisfy the main MCS criteria.
        merging_events (List[MergingEvent]): List of merging events recorded during tracking.
        splitting_events (List[SplittingEvent]): List of splitting events recorded during tracking.

    Returns:
        List[np.ndarray]: A new list of 2D arrays where any pixel with a track ID not in the
                          union of relevant IDs is set to 0.

    Usage:
        After running track_mcs(), suppose you obtain:

            mcs_ids_list, main_mcs_ids, lifetime_list, time_list, lat, lon,
            merging_events, splitting_events, tracking_centers_list = track_mcs(...)

        Then, to filter out ephemeral tracks that never merged or split into a main MCS, do:

            filtered_mcs_ids_list = filter_relevant_mcs(
                mcs_ids_list, main_mcs_ids, merging_events, splitting_events
            )
    """
    # Start with the main MCS IDs.
    relevant_ids = set(main_mcs_ids)

    # Add IDs involved in merging events.
    for event in merging_events:
        # Add all parent IDs.
        relevant_ids.update(event.parent_ids)
        # Also add the child ID.
        relevant_ids.add(event.child_id)

    # Add IDs involved in splitting events.
    for event in splitting_events:
        # Add the parent ID.
        relevant_ids.add(event.parent_id)
        # Add all child IDs.
        relevant_ids.update(event.child_ids)

    # Filter each timestep's array: only keep values in relevant_ids.
    filtered_mcs_ids_list = []
    for mcs_array in mcs_ids_list:
        filtered_array = mcs_array.copy()
        # Any pixel not in relevant_ids is set to 0.
        mask = ~np.isin(filtered_array, list(relevant_ids))
        filtered_array[mask] = 0
        filtered_mcs_ids_list.append(filtered_array)

    return filtered_mcs_ids_list


def filter_main_mcs(mcs_ids_list, main_mcs_ids):
    """
    Filters tracking results to include only the 'main' MCS IDs.

    Args:
        mcs_ids_list (List[numpy.ndarray]): List of 2D arrays with track IDs.
        main_mcs_ids (List[int]): List of track IDs considered 'main' MCS.

    Returns:
        List[numpy.ndarray]: A new list of arrays, where IDs not in main_mcs_ids are set to 0.
    """
    filtered_mcs_id_list = []
    for mcs_id_array in mcs_ids_list:
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
    Tracks MCSs across multiple timesteps using spatial overlap and stable ID assignment.

    Args:
        detection_results (List[dict]): Each dict contains:
            "final_labeled_regions" (2D array of labels),
            "center_points" (dict mapping label->(lat, lon)), optional
            "time" (datetime.datetime),
            "lat" (2D array),
            "lon" (2D array).
        main_lifetime_thresh (int): Minimum consecutive lifetime (in timesteps) for main MCS.
        main_area_thresh (float): Minimum area (km²) that must be reached in each timestep.
        grid_cell_area_km2 (float): Factor to convert pixel count to area (km²).
        nmaxmerge (int): Max allowed merging/splitting in a single timestep.

    Returns:
        Tuple:
            mcs_ids_list (List[numpy.ndarray]): Track ID arrays per timestep.
            main_mcs_ids (List[int]): List of IDs considered main MCS by end of tracking.
            lifetime_list (List[numpy.ndarray]): Per-timestep 2D arrays for pixel-wise lifetime.
            time_list (List[datetime.datetime]): Timestamps for each timestep.
            lat (numpy.ndarray): 2D lat array from the first timestep.
            lon (numpy.ndarray): 2D lon array from the first timestep.
            merging_events (List[MergingEvent]): All recorded merging events.
            splitting_events (List[SplittingEvent]): All recorded splitting events.
            tracking_centers_list (List[dict]): For each timestep, a dict mapping
                track_id -> (center_lat, center_lon).
    """
    previous_labeled_regions = None
    previous_cluster_ids = {}
    merge_split_cluster_ids = {}
    next_cluster_id = 1

    mcs_ids_list = []
    lifetime_list = []
    tracking_centers_list = []
    time_list = []
    lat = None
    lon = None

    lifetime_dict = defaultdict(int)
    max_area_dict = defaultdict(float)

    merging_events = []
    splitting_events = []

    for idx, detection_result in enumerate(detection_results):
        final_labeled_regions = detection_result["final_labeled_regions"]
        center_points_dict = detection_result.get("center_points", {})
        current_time = detection_result["time"]
        current_lat = detection_result["lat"]
        current_lon = detection_result["lon"]

        if lat is None:
            lat = current_lat
            lon = current_lon

        mcs_id = np.zeros_like(final_labeled_regions, dtype=np.int32)
        mcs_lifetime = np.zeros_like(final_labeled_regions, dtype=np.int32)

        unique_labels = np.unique(final_labeled_regions)
        unique_labels = unique_labels[unique_labels != 0]

        # If no valid clusters are detected, end all tracks for this timestep.
        if len(unique_labels) == 0:
            print(f"No clusters detected at {current_time}")
            previous_cluster_ids = {}
            previous_labeled_regions = None

            mcs_ids_list.append(mcs_id)
            lifetime_list.append(mcs_lifetime)
            time_list.append(current_time)
            tracking_centers_list.append({})
            continue

        # For the first timestep with clusters.
        if previous_labeled_regions is None:
            for label in unique_labels:
                cluster_mask = final_labeled_regions == label
                area = np.sum(cluster_mask) * grid_cell_area_km2
                assigned_id, next_cluster_id = assign_new_id(
                    label,
                    cluster_mask,
                    area,
                    next_cluster_id,
                    lifetime_dict,
                    max_area_dict,
                    mcs_id,
                    mcs_lifetime,
                )
                previous_cluster_ids[label] = assigned_id
                merge_split_cluster_ids[label] = assigned_id
        else:
            # For subsequent timesteps: check overlaps between previous and current clusters.
            overlap_map = check_overlaps(
                previous_labeled_regions,
                final_labeled_regions,
                previous_cluster_ids,
                overlap_threshold=10,
            )
            temp_assigned = {}
            labels_no_overlap = []

            for new_lbl, old_ids in overlap_map.items():
                if len(old_ids) == 0:
                    labels_no_overlap.append(new_lbl)
                elif len(old_ids) == 1:
                    chosen_id = old_ids[0]
                    # Continue the track.
                    handle_continuation(
                        new_label=new_lbl,
                        old_track_id=chosen_id,
                        final_labeled_regions=final_labeled_regions,
                        mcs_id=mcs_id,
                        mcs_lifetime=mcs_lifetime,
                        lifetime_dict=lifetime_dict,
                        max_area_dict=max_area_dict,
                        grid_cell_area_km2=grid_cell_area_km2,
                    )
                    temp_assigned[new_lbl] = chosen_id
                else:
                    # Merging scenario.
                    chosen_id = handle_merging(
                        new_label=new_lbl,
                        old_track_ids=old_ids,
                        merging_events=merging_events,
                        final_labeled_regions=final_labeled_regions,
                        current_time=current_time,
                        max_area_dict=max_area_dict,
                        grid_cell_area_km2=grid_cell_area_km2,
                        nmaxmerge=nmaxmerge,
                    )
                    mask = final_labeled_regions == new_lbl
                    mcs_id[mask] = chosen_id
                    lifetime_dict[chosen_id] += 1
                    temp_assigned[new_lbl] = chosen_id

            # Handle new clusters that had no overlap.
            new_assign_map, next_cluster_id = handle_no_overlap(
                labels_no_overlap,
                final_labeled_regions,
                next_cluster_id,
                lifetime_dict,
                max_area_dict,
                mcs_id,
                mcs_lifetime,
                grid_cell_area_km2,
            )
            temp_assigned.update(new_assign_map)

            # Check for splits: if one old track claims multiple new labels.
            oldid_to_newlist = defaultdict(list)
            for lbl, tid in temp_assigned.items():
                oldid_to_newlist[tid].append(lbl)

            for old_id, newlbls in oldid_to_newlist.items():
                if len(newlbls) > 1:
                    splitted_map, next_cluster_id = handle_splitting(
                        old_id,
                        newlbls,
                        final_labeled_regions,
                        current_time,
                        next_cluster_id,
                        splitting_events,
                        mcs_id,
                        mcs_lifetime,
                        lifetime_dict,
                        max_area_dict,
                        grid_cell_area_km2,
                        nmaxsplit=nmaxmerge,
                    )
                    for nl, finalid in splitted_map.items():
                        temp_assigned[nl] = finalid

            current_cluster_ids = temp_assigned
            previous_cluster_ids = current_cluster_ids

        previous_labeled_regions = final_labeled_regions.copy()

        mcs_ids_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        # Build a dictionary of track centers for this timestep.
        centers_this_timestep = {}
        label_by_cluster = defaultdict(list)
        for lbl, tid in previous_cluster_ids.items():
            label_by_cluster[tid].append(lbl)

        for tid, label_list in label_by_cluster.items():
            center_latlon = (None, None)
            for detect_label in label_list:
                detect_label_str = str(detect_label)
                if detect_label_str in center_points_dict:
                    center_latlon = center_points_dict[detect_label_str]
                    break
            centers_this_timestep[str(tid)] = center_latlon
        tracking_centers_list.append(centers_this_timestep)

    # ---- New Filtering Step ----
    # Instead of simply checking overall lifetime and maximum area,
    # we now require that each main MCS has at least main_lifetime_thresh consecutive
    # timesteps with an area >= main_area_thresh.
    #
    # For each track ID, we build a Boolean series for each timestep, where True means
    # the area for that track meets the main_area_thresh. Then we compute the maximum number
    # of consecutive True values. If that maximum is at least main_lifetime_thresh, the track qualifies.

    def compute_max_consecutive(bool_list):
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

    all_track_ids = list(lifetime_dict.keys())
    valid_ids = []
    for tid in all_track_ids:
        # Build a Boolean series: for each timestep, True if area for track tid >= main_area_thresh.
        bool_series = []
        for mcs_id_array in mcs_ids_list:
            area = np.sum(mcs_id_array == tid) * grid_cell_area_km2
            bool_series.append(area >= main_area_thresh)
        if compute_max_consecutive(bool_series) >= main_lifetime_thresh:
            valid_ids.append(tid)
    main_mcs_ids = valid_ids
    filtered_mcs_ids_list = filter_relevant_systems(
        mcs_ids_list, main_mcs_ids, merging_events, splitting_events
    )

    return (
        filtered_mcs_ids_list,
        main_mcs_ids,
        lifetime_list,
        time_list,
        lat,
        lon,
        merging_events,
        splitting_events,
        tracking_centers_list,
    )
