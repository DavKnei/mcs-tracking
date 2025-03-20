#!/usr/bin/env python
"""
tracking_main.py

Main routine for tracking Mesoscale Convective Systems (MCSs) across multiple timesteps.
Tracks are assigned via spatial overlap, and a robust filtering based on a lifting index (LI)
detection (provided in the detection results as 'lifting_index_regions') is applied.

The script returns per-timestep tracking arrays, main track IDs, lifetime arrays,
merging and splitting events, and tracking center positions.
"""

import numpy as np
import logging
from collections import defaultdict
from tracking_filter_func import filter_relevant_systems
from tracking_helper_func import (
    assign_new_id,
    check_overlaps,
    handle_continuation,
    handle_no_overlap,
    compute_max_consecutive,
    apply_robust_mask,
    build_tracking_centers,
)
from tracking_merging import handle_merging
from tracking_splitting import handle_splitting

logger = logging.getLogger(__name__)


def track_mcs(
    detection_results,
    main_lifetime_thresh,
    main_area_thresh,
    grid_cell_area_km2,
    nmaxmerge,
    use_li_filter=True,  # Flag to enable LI filtering if available
):
    """
    Tracks MCSs across multiple timesteps using spatial overlap and stable ID assignment.
    Additionally, if the detection_results include a variable "lifting_index_regions", then
    a track is considered robust only if at least one timestep in its lifetime satisfies the LI criteria.

    Args:
        detection_results (List[dict]): Each dict contains:
            - "final_labeled_regions" (2D array of labels),
            - "lifting_index_regions" (2D binary array; 1 indicates LI criterion met) [optional],
            - "center_points" (dict mapping label -> (lat, lon)) [optional],
            - "time" (datetime.datetime),
            - "lat" (2D array),
            - "lon" (2D array).
        main_lifetime_thresh (int): Minimum consecutive lifetime (in timesteps) for main MCS.
        main_area_thresh (float): Minimum area (km²) that must be reached in each timestep.
        grid_cell_area_km2 (float): Factor to convert pixel count to area (km²).
        nmaxmerge (int): Maximum allowed merging/splitting in a single timestep.
        use_li_filter (bool): If True and if "lifting_index_regions" exists in detection_results,
                              tracks that never meet the LI criterion (LI==1) are filtered out.

    Returns:
        Tuple:
            mcs_ids_list (List[np.ndarray]): Track ID arrays per timestep.
            main_mcs_ids (List[int]): List of IDs considered main MCS by end of tracking.
            lifetime_list (List[np.ndarray]): Per-timestep 2D arrays for pixel-wise lifetime.
            time_list (List[datetime.datetime]): Timestamps for each timestep.
            lat (np.ndarray): 2D lat array from the first timestep.
            lon (np.ndarray): 2D lon array from the first timestep.
            merging_events (List): All recorded merging events.
            splitting_events (List): All recorded splitting events.
            tracking_centers_list (List[dict]): For each timestep, dict mapping track_id -> (center_lat, center_lon).
            main_mcs_ids_robust (List[np.ndarray]): Same as mcs_ids_list but with non-robust tracks removed.
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

    # Dictionary to track robust flag for each assigned track ID.
    robust_flag_dict = {}

    # Determine if LI filtering is available (only need to check detection_results[0])
    use_li = use_li_filter and ("lifting_index_regions" in detection_results[0])

    # Optionally limit number of timesteps for testing:
    detection_results = detection_results[:70]

    for idx, detection_result in enumerate(detection_results):
        final_labeled_regions = detection_result["final_labeled_regions"]
        center_points_dict = detection_result.get("center_points", {})
        current_time = detection_result["time"]
        current_lat = detection_result["lat"]
        current_lon = detection_result["lon"]

        # Get LI regions if available.
        if use_li:
            li_regions = detection_result["lifting_index_regions"]
        else:
            li_regions = None

        # Set spatial coordinates on first timestep.
        if lat is None:
            lat = current_lat
            lon = current_lon

        # Initialize ID and lifetime arrays for current timestep.
        mcs_id = np.zeros_like(final_labeled_regions, dtype=np.int32)
        mcs_lifetime = np.zeros_like(final_labeled_regions, dtype=np.int32)

        unique_labels = np.unique(final_labeled_regions)
        unique_labels = unique_labels[unique_labels != 0]

        if len(unique_labels) == 0:
            logger.info(f"No clusters detected at {current_time}")
            previous_cluster_ids = {}
            previous_labeled_regions = None

            mcs_ids_list.append(mcs_id)
            lifetime_list.append(mcs_lifetime)
            time_list.append(current_time)
            tracking_centers_list.append({})
            continue

        if previous_labeled_regions is None:
            # First timestep with clusters: assign new track IDs.
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
                if use_li:
                    is_convective = np.all(li_regions[cluster_mask] == 1)
                else:
                    is_convective = True
                robust_flag_dict[assigned_id] = is_convective
        else:
            # Subsequent timesteps: check overlaps between previous and current clusters.
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
                    if use_li:
                        current_mask = final_labeled_regions == new_lbl
                        current_convective = np.all(li_regions[current_mask] == 1)
                    else:
                        current_convective = True
                    robust_flag_dict[chosen_id] = (
                        robust_flag_dict.get(chosen_id, False) or current_convective
                    )
                else:
                    # Merging: handle multiple overlapping previous clusters.
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
                    if use_li:
                        current_mask = final_labeled_regions == new_lbl
                        current_convective = np.all(li_regions[current_mask] == 1)
                    else:
                        current_convective = True
                    robust_flag = (
                        any(robust_flag_dict.get(old_id, False) for old_id in old_ids)
                        or current_convective
                    )
                    robust_flag_dict[chosen_id] = robust_flag

            # Handle clusters with no overlap.
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
            # Handle splitting events.
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
                        # Re-evaluate the LI for the child region.
                        current_mask = final_labeled_regions == nl
                        is_convective = np.all(li_regions[current_mask] == 1)
                        robust_flag_dict[finalid] = is_convective
                        logger.info(
                            f"Track splitting at {current_time} for parent track {old_id}. "
                            f"New child track {finalid} assigned robust flag: {robust_flag_dict[finalid]}"
                        )
            current_cluster_ids = temp_assigned
            previous_cluster_ids = current_cluster_ids
            logger.info(f"MCS tracking at {current_time} processed.")

        previous_labeled_regions = final_labeled_regions.copy()

        mcs_ids_list.append(mcs_id)
        lifetime_list.append(mcs_lifetime)
        time_list.append(current_time)

        # Build tracking centers for this timestep.
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

    # ---- Final Filtering Step Based on Lifetime and Area (unchanged) ----
    valid_ids = []
    for tid in list(lifetime_dict.keys()):
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

    # ---- Apply LI robust filtering: remove tracks that never met the LI criterion.
    if use_li:
        all_ids = np.unique(
            np.concatenate([arr.ravel() for arr in filtered_mcs_ids_list])
        )
        all_ids = all_ids[all_ids != 0]
        main_mcs_ids_robust = [
            apply_robust_mask(arr, robust_flag_dict) for arr in filtered_mcs_ids_list
        ]
    else:
        main_mcs_ids_robust = filtered_mcs_ids_list

    logger.info(f"Tracking finished. {len(main_mcs_ids)} main MCSs found.")
    return (
        main_mcs_ids_robust,
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
