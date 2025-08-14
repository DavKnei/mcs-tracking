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
from tracking_filter_func import filter_main_mcs, filter_relevant_systems
from tracking_helper_func import (
    assign_new_id,
    check_overlaps,
    handle_continuation,
    handle_no_overlap,
    compute_max_consecutive,
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
    use_li_filter
):
    """
    Tracks Mesoscale Convective Systems (MCSs) and filters them based on a combined set of criteria.

    This function first tracks all detected precipitation features over time using spatial overlap,
    handling complex merging and splitting events. After the initial tracking, it performs a
    rigorous filtering step to identify "main MCSs". A track qualifies as a main MCS only if it
    contains a continuous period of at least 'main_lifetime_thresh' hours where, simultaneously,
    its area is greater than 'main_area_thresh' and it is in a convective environment (if 'use_li_filter' is True).

    The function returns three distinct sets of track IDs representing different levels of filtering,
    from the most restrictive ("in-phase" MCSs) to the most inclusive ("full family tree").

    Args:
        detection_results (List[dict]): A list where each dictionary represents one timestep and contains:
            - "final_labeled_regions" (np.ndarray): 2D array of detected cluster labels.
            - "lifting_index_regions" (np.ndarray): 2D binary array where 1 indicates a cluster met the LI criterion. Optional, used if 'use_li_filter' is True.
            - "center_points" (dict): Mapping of cluster label to its (lat, lon) center. Optional.
            - "time" (datetime.datetime): Timestamp for the data.
            - "lat" (np.ndarray): 2D array of latitudes.
            - "lon" (np.ndarray): 2D array of longitudes.
        main_lifetime_thresh (int): The minimum number of consecutive hours a track must simultaneously meet the area and LI criteria to be considered a main MCS.
        main_area_thresh (float): The minimum area (in km²) a track must have to be considered in its mature phase.
        grid_cell_area_km2 (float): The area of a single grid cell in km².
        nmaxmerge (int): The maximum number of parent systems to consider in a single merging event.
        use_li_filter (bool): If True, enables the convective environment check based on the "lifting_index_regions" data.

    Returns:
        Tuple: A tuple containing the following organized results:
            - robust_mcs_id (List[np.ndarray]): The most restrictive output. Contains track IDs only for the timesteps where the system is **simultaneously** larger than 'main_area_thresh' AND meets the convective LI criteria. This isolates the mature, "in-phase" portion of the MCSs.
            - main_mcs_id (List[np.ndarray]): Shows the **full lifetime** of all tracks that were identified as main MCSs. This includes their formation and dissipation stages where they may not meet the area or LI criteria.
            - main_mcs_id_merge_split (List[np.ndarray]): The most inclusive output. Shows the **full "family tree"**, containing the full lifetime of main MCSs plus the full lifetime of all smaller systems that merged into or split from them.
            - lifetime_list (List[np.ndarray]): A list of 2D arrays showing the pixel-wise lifetime (in timesteps) of all tracked clusters.
            - time_list (List[datetime.datetime]): A list of the timestamps corresponding to each frame.
            - lat (np.ndarray): A 2D array of latitude values.
            - lon (np.ndarray): A 2D array of longitude values.
            - merging_events (List[MergingEvent]): A list of all recorded merging events.
            - splitting_events (List[SplittingEvent]): A list of all recorded splitting events.
            - tracking_centers_list (List[dict]): A list of dictionaries, one for each timestep, mapping track IDs to their (lat, lon) center points."""
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
    convective_history = defaultdict(dict)

    # Determine if LI filtering is available (only need to check detection_results[0])
    use_li = use_li_filter and ("lifting_index_regions" in detection_results[0])

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
                convective_history[assigned_id][idx] = is_convective
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
                    convective_history[chosen_id][idx] = current_convective
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
                    convective_history[chosen_id][idx] = current_convective
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

    # ---- Final Filtering Step ----
    # A track is a "main MCS" if it has a continuous period of at least
    # 'main_lifetime_thresh' where BOTH area and LI criteria are met.

    mcs_ids = []
    # Iterate through every track that ever existed
    for tid in list(lifetime_dict.keys()):

        # For each track, create a boolean series indicating when it meets BOTH criteria
        bool_series_combined = []
        for i, mcs_id_array in enumerate(mcs_ids_list):
            area = np.sum(mcs_id_array == tid) * grid_cell_area_km2
            meets_area_criteria = area >= main_area_thresh

            # Check the convective history for this timestep
            # If not using LI, all systems are considered convective
            meets_li_criteria = (
                convective_history[tid].get(i, False) if use_li else True
            )

            # Both must be true for the system to be in a robust state at this hour
            bool_series_combined.append(meets_area_criteria and meets_li_criteria)

        # Now, find the longest continuous period of this robust state
        if compute_max_consecutive(bool_series_combined) >= main_lifetime_thresh:
            mcs_ids.append(tid)

    logger.info(
        f"Tracking identified {len(mcs_ids)} main MCSs after combined filtering."
    )

    # --- Generate the 3 Final Output Variables ---

    # Output 3 (Most Inclusive): Full "Family Tree"
    mcs_id_merge_split = filter_relevant_systems(
        mcs_ids_list, mcs_ids, merging_events, splitting_events
    )

    # Output 2 (Intermediate): Full Lifetime of Main MCSs
    mcs_id = filter_main_mcs(mcs_ids_list, mcs_ids)

    # Output 1 (Most Restrictive): "In-Phase" MCSs
    robust_mcs_id = []
    for i, mcs_id_array in enumerate(mcs_id):
        frame_in_phase = mcs_id_array.copy()
        unique_ids_in_frame = np.unique(frame_in_phase[frame_in_phase > 0])

        for tid in unique_ids_in_frame:
            area = np.sum(mcs_id_array == tid) * grid_cell_area_km2

            # Get the convective status for this track at this timestep
            is_convective_this_step = (
                convective_history[tid].get(i, False) if use_li else True
            )

            # A track is "in-phase" ONLY if it's both large enough AND convective
            if area < main_area_thresh or not is_convective_this_step:
                frame_in_phase[frame_in_phase == tid] = 0
        robust_mcs_id.append(frame_in_phase)

    # Final return statement
    return (
        robust_mcs_id,
        mcs_id,
        mcs_id_merge_split,
        lifetime_list,
        time_list,
        lat,
        lon,
        merging_events,
        splitting_events,
        tracking_centers_list,
    )
