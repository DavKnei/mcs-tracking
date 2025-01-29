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
        if max_area_dict.get(pid, 0) > best_area:
            best_area = max_area_dict[pid]
            best_id = pid
    return best_id


def check_overlaps(
    previous_labeled_regions,
    final_labeled_regions,
    previous_cluster_ids,
    overlap_threshold=10
):
    """
    Checks overlap between the old labeled regions (previous_labeled_regions)
    and the new labeled regions (final_labeled_regions), building a mapping
    new_label -> list of old cluster IDs that overlap above overlap_threshold%.

    Parameters
    ----------
    previous_labeled_regions : 2D int array
        Labeled regions from the previous timestep's detection.
        (labels, not track IDs; -1 means no cluster.)
    final_labeled_regions : 2D int array
        Labeled regions from the current timestep's detection.
        (labels, not track IDs; -1 means no cluster.)
    previous_cluster_ids : dict
        Maps old detection labels to old track IDs. e.g. { old_label: track_id }
    overlap_threshold : float
        Minimum overlap percentage for an old cluster ID to be considered relevant.

    Returns
    -------
    overlap_map : dict
        A dict: { new_label (int) : list of old track IDs (int) }
        If no old cluster ID meets overlap_threshold, it will be an empty list.

    Notes
    -----
    - We are not returning a single 'best' ID here. We gather *all* old IDs above threshold,
      letting us detect merges (multiple old IDs overlap one new label) or no overlap (0 old IDs).
    """

    overlap_map = {}  # new_label -> list of old track IDs

    unique_prev_labels = np.unique(previous_labeled_regions)
    unique_prev_labels = unique_prev_labels[unique_prev_labels != -1]
    unique_curr_labels = np.unique(final_labeled_regions)
    unique_curr_labels = unique_curr_labels[unique_curr_labels != -1]

    for new_label in unique_curr_labels:
        curr_mask = (final_labeled_regions == new_label)
        curr_area = np.sum(curr_mask)

        # We'll gather relevant old track IDs here
        relevant_old_ids = []

        if curr_area == 0:
            overlap_map[new_label] = relevant_old_ids
            continue

        for old_label_detection in unique_prev_labels:
            # This old_label_detection is a label from the previous DETECTION
            old_track_id = previous_cluster_ids.get(old_label_detection, None)
            if old_track_id is None:
                # Possibly we have a label not in previous_cluster_ids => no assigned track
                continue

            prev_mask = (previous_labeled_regions == old_label_detection)
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
    grid_cell_area_km2
):
    """
    Assign brand-new track IDs to all new labels that have no overlap
    with any previous cluster IDs.

    Parameters
    ----------
    new_labels_no_overlap : list of int
        All new detection labels that had no old cluster ID.
    final_labeled_regions : 2D array
        Current detection labels (for area computation).
    next_cluster_id : int
        Next available track ID for new ephemeral IDs.
    lifetime_dict, max_area_dict : dict
        For tracking lifetime and max area.
    mcs_id, mcs_lifetime : 2D arrays
        Arrays to write assigned IDs / lifetime.
    grid_cell_area_km2 : float
        Grid cell area (for area computation).

    Returns
    -------
    assigned_ids_map : dict
        Mapping new_label -> assigned track ID
    next_cluster_id : int
        Possibly incremented
    """
    assigned_ids_map = {}

    for lbl in new_labels_no_overlap:
        mask = (final_labeled_regions == lbl)
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
    grid_cell_area_km2
):
    """Continues an existing old_track_id for the new_label cluster.

    Parameters
    ----------
    new_label : int
        New detection label.
    old_track_id : int
        Old track ID to continue.
    final_labeled_regions : 2D array
        Current detection labels.
    mcs_id, mcs_lifetime : 2D arrays
        Track ID and lifetime arrays.
    lifetime_dict, max_area_dict : dict
        For tracking lifetime and max area.
    grid_cell_area_km2 : float
    """

    mask = (final_labeled_regions == new_label)
    area_pixels = np.sum(mask)
    area_km2 = area_pixels * grid_cell_area_km2

    mcs_id[mask] = old_track_id
    lifetime_dict[old_track_id] += 1
    mcs_lifetime[mask] = lifetime_dict[old_track_id]
    if area_km2 > max_area_dict[old_track_id]:
        max_area_dict[old_track_id] = area_km2

def handle_merging(
    new_label,
    old_track_ids,  # list of old track IDs
    merging_events,
    final_labeled_regions,
    current_time,
    max_area_dict,
    grid_cell_area_km2,
    nmaxmerge=5
):
    """
    Multiple old track IDs => merges into the new_label's final track ID.
    We'll pick a 'dominant' old ID (largest area or first in list).
    Then unify new_label => that ID, record the event, etc.

    Returns the chosen old track ID for this new_label.
    """
    if len(old_track_ids) == 1:
        return old_track_ids[0]  # no real merge

    # pick largest area among old_track_ids
    best_id = get_dominant_cluster(old_track_ids, max_area_dict)

    # if more than nmaxmerge, limit
    if len(old_track_ids) > nmaxmerge:
        old_track_ids = old_track_ids[:nmaxmerge]

    # record MergingEvent if merging_events is not None
    if merging_events is not None and len(old_track_ids) > 1:
        from datetime import datetime
        # compute child_area from new_label
        mask = (final_labeled_regions == new_label)
        area_pix = np.sum(mask)
        child_area = area_pix * grid_cell_area_km2
        parent_areas = [max_area_dict[pid] for pid in old_track_ids]
        mevt = MergingEvent(
            time=current_time,
            parent_ids=old_track_ids,
            child_id=best_id,
            parent_areas=parent_areas,
            child_area=child_area
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
    nmaxsplit=5
):
    """
    One old_track_id => multiple new labels => splitting.
    The largest new label keeps old_track_id, others get new ephemeral IDs.
    """
    if len(new_label_list) <= 1:
        return {}, next_cluster_id  # no actual split

    # measure areas of new labels
    new_label_areas = []
    for nlbl in new_label_list:
        mask = (final_labeled_regions == nlbl)
        area_pix = np.sum(mask)
        new_label_areas.append(area_pix * grid_cell_area_km2)

    # pick largest
    idx_sorted = sorted(range(len(new_label_list)),
                        key=lambda i: new_label_areas[i],
                        reverse=True)
    keep_idx = idx_sorted[0]
    keep_label = new_label_list[keep_idx]
    keep_area = new_label_areas[keep_idx]

    # the rest => new IDs
    splitted_assign_map = {}
    
    # unify keep_label => old_track_id
    splitted_assign_map[keep_label] = old_track_id
    lifetime_dict[old_track_id] -= 1  #  TODO: Weird error in splitting lifetime is 2 to high. lifetime_dict only gets updated +1 in handle_continuation. This fixes it for now.
    if keep_area > max_area_dict[old_track_id]:
        max_area_dict[old_track_id] = keep_area

    # rewriting in the array
    keep_mask = (final_labeled_regions == keep_label)
    mcs_id[keep_mask] = old_track_id
    mcs_lifetime[keep_mask] = lifetime_dict[old_track_id]
    
    splitted_child_labels = []
    splitted_child_areas = []

    # for all other new labels
    for i in idx_sorted[1:]:
        lbl = new_label_list[i]
        area_s = new_label_areas[i]
        # new ephemeral ID
        splitted_assign_map[lbl] = next_cluster_id
        lifetime_dict[next_cluster_id] = 1
        max_area_dict[next_cluster_id] = area_s
        # rewrite
        mask_s = (final_labeled_regions == lbl)
        mcs_id[mask_s] = next_cluster_id
        mcs_lifetime[mask_s] = 1
        splitted_child_labels.append(next_cluster_id)
        splitted_child_areas.append(area_s)
        next_cluster_id += 1
    
    # record a SplittingEvent if we have more than 1 new label
    if splitting_events is not None and len(new_label_list) > 1:
        if len(new_label_list) > nmaxsplit:
            new_label_list = new_label_list[:nmaxsplit]
        sevt = SplittingEvent(
            time=current_time,
            parent_id=old_track_id,
            child_ids=[splitted_assign_map[lbl] for lbl in new_label_list],
            parent_area=max_area_dict[old_track_id],
            child_areas=[max_area_dict[splitted_assign_map[lbl]] for lbl in new_label_list],
        )
        splitting_events.append(sevt)

    return splitted_assign_map, next_cluster_id

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

    previous_labeled_regions = None
    previous_cluster_ids = {}
    merge_split_cluster_ids = {}
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

        # Identify how many unique labels > -1 exist
        unique_labels = np.unique(final_labeled_regions)
        unique_labels = unique_labels[unique_labels != -1]

        # If no valid clusters in this timestep => all -1 => end current tracks
        if len(unique_labels) == 0:
            print(f"No clusters detected at {current_time}")
            # End all active tracks
            previous_cluster_ids = {}
            previous_labeled_regions = None  # or keep it zero as well

            # We already have zeros in mcs_detected, mcs_id, mcs_lifetime
            # Just append them to the final output lists and continue
            mcs_detected_list.append(mcs_detected)
            mcs_id_list.append(mcs_id)
            lifetime_list.append(mcs_lifetime)
            time_list.append(current_time)
            continue

        # -------------------------
        # If we do have clusters, proceed:
        if previous_labeled_regions is None:
            # First timestep (with actual clusters)
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
                merge_split_cluster_ids[label] = assigned_id  #  Assign all clusters to themselves in the first timestep
                mcs_detected[cluster_mask] = 1
        else:
            # Subsequent timesteps
            # PSEUDOCODE for the "subsequent timesteps" part in track_mcs:

            # (1) compute overlap_map
            overlap_map = check_overlaps(
                previous_labeled_regions,
                final_labeled_regions,
                previous_cluster_ids,
                overlap_threshold=10
            )

            temp_assigned = {}  # new_label -> chosen ID (merging/continuation/new)
            labels_no_overlap = []
            for new_lbl, old_ids in overlap_map.items():
                if len(old_ids) == 0:
                    labels_no_overlap.append(new_lbl)
                elif len(old_ids) == 1:
                    # continuation
                    chosen_id = old_ids[0]
                    handle_continuation(
                        new_label=new_lbl,
                        old_track_id=chosen_id,
                        final_labeled_regions=final_labeled_regions,
                        mcs_id=mcs_id,
                        mcs_lifetime=mcs_lifetime,
                        lifetime_dict=lifetime_dict,
                        max_area_dict=max_area_dict,
                        grid_cell_area_km2=grid_cell_area_km2
                    )
                    temp_assigned[new_lbl] = chosen_id
                else:
                    # merging
                    chosen_id = handle_merging(
                        new_label=new_lbl,
                        old_track_ids=old_ids,
                        merging_events=merging_events,
                        final_labeled_regions=final_labeled_regions,
                        current_time=current_time,
                        mcs_id=mcs_id,
                        mcs_lifetime=mcs_lifetime,
                        lifetime_dict=lifetime_dict,
                        max_area_dict=max_area_dict,
                        grid_cell_area_km2=grid_cell_area_km2,
                        nmaxmerge=nmaxmerge
                    )
                    temp_assigned[new_lbl] = chosen_id

                # handle no-overlap labels => new ephemeral IDs
                new_assign_map, next_cluster_id = handle_no_overlap(
                    labels_no_overlap,
                    final_labeled_regions,
                    next_cluster_id,
                    lifetime_dict,
                    max_area_dict,
                    mcs_id,
                    mcs_lifetime,
                    grid_cell_area_km2
                )
                temp_assigned.update(new_assign_map)

                # now 'temp_assigned' might have multiple new labels => the same old ID => splitting
                # invert this to old_id -> list of new labels
                oldid_map = defaultdict(list)
                for lbl, id_val in temp_assigned.items():
                    oldid_map[id_val].append(lbl)

                # check if any old_id is used by multiple new labels => splitting
                for old_id, new_lbls in oldid_map.items():
                    if len(new_lbls) > 1:
                        # immediate splitting
                        splitted_map, next_cluster_id = handle_splitting(
                            old_track_id=old_id,
                            new_label_list=new_lbls,
                            final_labeled_regions=final_labeled_regions,
                            current_time=current_time,
                            next_cluster_id=next_cluster_id,
                            splitting_events=splitting_events,
                            mcs_id=mcs_id,
                            mcs_lifetime=mcs_lifetime,
                            lifetime_dict=lifetime_dict,
                            max_area_dict=max_area_dict,
                            grid_cell_area_km2=grid_cell_area_km2,
                            nmaxsplit=nmaxmerge
                        )
                        # splitted_map : { new_lbl : final ID }
                        # update temp_assigned with splitted_map
                        for nl, final_id in splitted_map.items():
                            temp_assigned[nl] = final_id

                        

                # finalize current_cluster_ids => { new_label : final track ID }
                current_cluster_ids = temp_assigned
                previous_cluster_ids = current_cluster_ids

            
        previous_labeled_regions = final_labeled_regions.copy()
        
        # Append results
        mcs_detected_list.append(mcs_detected)
        mcs_id_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        total_lifetime_dict = lifetime_dict
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
